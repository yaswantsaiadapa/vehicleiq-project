"""
VehicleIQ — DetectorService
=============================
Loads YOLO26n once, exposes a single `process_video` method that
runs YOLO + ByteTrack over every frame and returns a pandas DataFrame.
The tracker state is reset between videos so IDs never bleed across.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from app import config

log = logging.getLogger(__name__)


class DetectorService:
    """
    YOLO + ByteTrack vehicle detector.

    Models are loaded once at instantiation; the caller passes frames
    or a video path and receives a structured DataFrame of detections.
    """

    def __init__(self) -> None:
        from ultralytics import YOLO  # deferred import — slow to load

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "Loading YOLO model: %s  (device=%s)", config.YOLO_MODEL_NAME, self.device
        )
        self.model = YOLO(config.YOLO_MODEL_NAME)
        log.info("YOLO model ready")

        if self.device == "cpu":
            log.warning(
                "GPU not available — YOLO will run on CPU.  "
                "Expect ~15-20 minutes for a 2000-frame video."
            )

    # ── Public API ────────────────────────────────────────────────

    def process_video(
        self,
        video_path: Path,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Run YOLO + ByteTrack on every frame of *video_path*.

        Parameters
        ----------
        video_path:
            Path to the video file.
        progress_callback:
            Optional callable(frames_done, total_frames) for status updates.

        Returns
        -------
        pd.DataFrame with columns:
            frame_number, track_id, class_id, class_name,
            confidence, x1, y1, x2, y2
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        log.info(
            "Starting tracking: %s  frames=%d  fps=%.1f",
            video_path.name,
            total_frames,
            fps,
        )

        # Reset tracker between videos so IDs start fresh
        self._reset_tracker()

        records: list[dict] = []
        frame_number = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_number += 1
                detections = self._track_frame(frame, frame_number)
                records.extend(detections)

                if progress_callback and frame_number % 50 == 0:
                    progress_callback(frame_number, total_frames)

        finally:
            cap.release()

        log.info(
            "Tracking complete: %d frames  %d detections",
            frame_number,
            len(records),
        )

        if not records:
            return pd.DataFrame(
                columns=[
                    "frame_number", "track_id", "class_id", "class_name",
                    "confidence", "x1", "y1", "x2", "y2",
                ]
            )

        return pd.DataFrame(records)

    def get_video_info(self, video_path: Path) -> dict:
        """Return basic metadata for a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return info

    # ── Internal ──────────────────────────────────────────────────

    def _reset_tracker(self) -> None:
        """Clear ByteTrack state so track IDs start from 1 for each video."""
        try:
            if (
                hasattr(self.model, "predictor")
                and self.model.predictor is not None
                and hasattr(self.model.predictor, "trackers")
                and self.model.predictor.trackers
            ):
                for tracker in self.model.predictor.trackers:
                    tracker.reset()
        except Exception as e:
            log.debug("Tracker reset skipped: %s", e)

    def _track_frame(self, frame: np.ndarray, frame_number: int) -> list[dict]:
        """
        Run YOLO + ByteTrack on one frame.
        Returns a list of detection dicts (may be empty).
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=config.VEHICLE_CLASSES,
            conf=config.YOLO_CONFIDENCE,
            imgsz=config.YOLO_IMGSZ,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        track_ids = result.boxes.id.detach().cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()

        detections = []
        for track_id, box, class_id, conf in zip(track_ids, boxes, class_ids, confidences):
            x1, y1, x2, y2 = box
            detections.append(
                {
                    "frame_number": frame_number,
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "class_name": config.VEHICLE_CLASS_IDS.get(int(class_id), str(class_id)),
                    "confidence": float(conf),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )
        return detections
