"""
VehicleIQ — VideoProcessorService
====================================
Orchestrates the full processing pipeline for a single uploaded video:

  1. Read video metadata (fps, resolution, frame count)
  2. YOLO + ByteTrack → per-frame detection DataFrame          (DetectorService)
  3. Compute image-space velocity per track
  4. Temporal crop sampling (≤8 crops per track, linspace)
  5. Extract crops from video (second seek-based pass)
  6. SigLIP2 image encoding + mean-pooling                     (EmbeddingService)
  7. Zero-shot attribute classification                         (AttributeService)
  8. Assemble vehicle_record JSON
  9. Insert into SQLite + FAISS                                 (DatabaseService, RetrievalService)
  10. Persist FAISS index
  11. Mark video status = 'complete'

Runs entirely in a background thread — never blocks the event loop.
Status updates are written to the `videos` table so the frontend
can poll without freezing.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from app import config
from app.services.attribute import AttributeService
from app.services.database import DatabaseService
from app.services.detector import DetectorService
from app.services.embedding import EmbeddingService
from app.services.retrieval import RetrievalService

log = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoProcessorService:
    """
    Runs the full VehicleIQ pipeline on an uploaded video.

    Call `process_video` from a background thread / executor — it is
    a synchronous, blocking call that may run for several minutes on CPU.
    """

    def __init__(
        self,
        detector: DetectorService,
        embedding: EmbeddingService,
        attribute: AttributeService,
        db: DatabaseService,
        retrieval: RetrievalService,
    ) -> None:
        self._detector = detector
        self._emb = embedding
        self._attr = attribute
        self._db = db
        self._retrieval = retrieval

    # ── Public entry point ────────────────────────────────────────

    def generate_video_id(self) -> str:
        """Generate a short, unique video identifier."""
        return "vid_" + uuid.uuid4().hex[:10]

    def process_video(self, video_id: str, video_path: Path) -> None:
        """
        Full pipeline for one video.  Designed to be called from a
        background thread via asyncio.get_event_loop().run_in_executor().

        Status transitions written to DB:
            queued → processing → complete | failed
        """
        log.info("Pipeline start: video_id=%s  path=%s", video_id, video_path)

        try:
            self._run_pipeline(video_id, video_path)
        except Exception as exc:
            log.exception("Pipeline failed for %s: %s", video_id, exc)
            self._db.update_video_status(
                video_id,
                "failed",
                progress_message=str(exc),
                error_message=str(exc),
            )

    # ── Pipeline steps ────────────────────────────────────────────

    def _run_pipeline(self, video_id: str, video_path: Path) -> None:
        # ── Step 1: Video metadata ────────────────────────────────
        info = self._detector.get_video_info(video_path)
        fps: float = info["fps"] or 10.0  # fallback to 10 fps if unreadable
        frame_count: int = info["frame_count"]
        width: int = info["width"]
        height: int = info["height"]
        crop_base_dir = str(config.CROPS_DIR / video_id)

        self._db.update_video_status(video_id, "processing", "Detecting vehicles…")

        # ── Step 2: YOLO + ByteTrack ──────────────────────────────
        def _progress(done: int, total: int) -> None:
            self._db.update_video_status(
                video_id,
                "processing",
                f"Detecting vehicles… {done}/{total} frames",
            )

        tracks_df = self._detector.process_video(video_path, progress_callback=_progress)

        if tracks_df.empty:
            self._db.update_video_status(
                video_id, "complete",
                progress_message="No vehicles detected",
                vehicle_count=0,
            )
            self._retrieval.save()
            return

        log.info(
            "Tracking done: %d unique tracks, %d detections",
            tracks_df["track_id"].nunique(),
            len(tracks_df),
        )

        # ── Step 3: Velocity per track ────────────────────────────
        self._db.update_video_status(video_id, "processing", "Computing velocity…")
        tracks_df = _compute_velocity(tracks_df, fps)

        # ── Step 4 & 5: Crop extraction ───────────────────────────
        self._db.update_video_status(
            video_id, "processing", "Extracting representative crops…"
        )
        crop_records = _extract_crops(
            video_path=video_path,
            tracks_df=tracks_df,
            crops_root=Path(crop_base_dir),
            max_crops=config.MAX_TEMPORAL_CROPS,
        )  # dict: {track_id: [{"frame_number": N, "crop_path": Path}]}

        # ── Steps 6–9: Encode, classify, store ───────────────────
        self._db.update_video_status(
            video_id, "processing", "Generating embeddings…"
        )
        vehicle_count = self._encode_and_store(
            video_id=video_id,
            tracks_df=tracks_df,
            crop_records=crop_records,
            crop_base_dir=crop_base_dir,
        )

        # ── Step 10: Persist FAISS ────────────────────────────────
        self._db.update_video_status(
            video_id, "processing", "Saving search index…"
        )
        self._retrieval.save()

        # ── Step 11: Mark complete ────────────────────────────────
        self._db.update_video_status(
            video_id,
            "complete",
            progress_message=f"Done — {vehicle_count} vehicles indexed",
            vehicle_count=vehicle_count,
        )
        log.info(
            "Pipeline complete: video_id=%s  vehicles=%d",
            video_id,
            vehicle_count,
        )

    # ── Encode + store ────────────────────────────────────────────

    def _encode_and_store(
        self,
        video_id: str,
        tracks_df: pd.DataFrame,
        crop_records: dict,
        crop_base_dir: str,
    ) -> int:
        """Encode each track and persist to SQLite + FAISS."""
        vehicle_count = 0

        for track_id, group in tracks_df.groupby("track_id"):
            group = group.sort_values("frame_number")

            # ── Crop paths ────────────────────────────────────────
            crop_items = crop_records.get(int(track_id), [])
            crop_paths = [item["crop_path"] for item in crop_items]
            relative_crops = [
                str(p.relative_to(crop_base_dir)).replace("\\", "/")
                for p in crop_paths
            ]

            # ── Load images and weights ───────────────────────────
            images: list[Image.Image] = []
            weights: list[float] = []
            for item in crop_items:
                p = item["crop_path"]
                if p.exists():
                    try:
                        images.append(Image.open(p).convert("RGB"))
                        weights.append(item.get("weight", 1.0))
                    except Exception as e:
                        log.warning("Cannot open crop %s: %s", p, e)

            if not images:
                log.debug("Track %d: no valid crops — skipping", track_id)
                continue

            # ── Quality-weighted vehicle embedding ────────────────
            vehicle_embedding = self._emb.encode_vehicle(images, weights=weights)
            if vehicle_embedding is None:
                continue

            # ── Attributes ────────────────────────────────────────
            attrs = self._attr.classify(vehicle_embedding)

            # ── Vehicle type from YOLO class distribution ─────────
            class_dist = group["class_name"].value_counts().to_dict()
            vehicle_type = max(class_dist, key=class_dist.get)
            type_score = float(group["confidence"].mean())

            # ── Velocity summary ──────────────────────────────────
            valid_speed = group["speed_pixels_per_sec"].dropna()
            mean_speed = float(valid_speed.mean()) if len(valid_speed) else None
            median_speed = float(valid_speed.median()) if len(valid_speed) else None
            max_speed = float(valid_speed.max()) if len(valid_speed) else None

            # ── Unique vehicle ID ─────────────────────────────────
            vehicle_id = f"{video_id}_v_{int(track_id):05d}"

            # ── Add to FAISS ──────────────────────────────────────
            embedding_index = self._retrieval.add_vehicle(
                vehicle_id, vehicle_embedding
            )

            # ── Assemble vehicle_record JSON ──────────────────────
            record = {
                "vehicle_id": vehicle_id,
                "source": {
                    "video_id": video_id,
                },
                "tracking": {
                    "track_id": int(track_id),
                    "first_frame": int(group["frame_number"].min()),
                    "last_frame": int(group["frame_number"].max()),
                    "frame_count": int(group["frame_number"].nunique()),
                },
                "vehicle": {
                    "type": {
                        "value": vehicle_type,
                        "score": round(type_score, 6),
                    },
                    "color": attrs.get("color", {}),
                    "body_style": attrs.get("body_style", {}),
                    "orientation": attrs.get("orientation", {}),
                },
                "velocity": {
                    "coordinate_system": "image",
                    "unit": "pixels_per_second",
                    "mean_speed": round(mean_speed, 4) if mean_speed is not None else None,
                    "median_speed": round(median_speed, 4) if median_speed is not None else None,
                    "max_speed": round(max_speed, 4) if max_speed is not None else None,
                },
                "visual": {
                    "representative_crops": relative_crops,
                },
            }

            # ── Persist to SQLite ─────────────────────────────────
            self._db.insert_vehicle(
                vehicle_id=vehicle_id,
                video_id=video_id,
                track_id=int(track_id),
                first_frame=int(group["frame_number"].min()),
                last_frame=int(group["frame_number"].max()),
                frame_count=int(group["frame_number"].nunique()),
                vehicle_type=vehicle_type,
                vehicle_type_score=type_score,
                color=attrs["color"]["value"],
                color_score=attrs["color"]["score"],
                body_style=attrs["body_style"]["value"],
                body_style_score=attrs["body_style"]["score"],
                orientation=attrs["orientation"]["value"],
                orientation_score=attrs["orientation"]["score"],
                mean_speed=mean_speed,
                median_speed=median_speed,
                max_speed=max_speed,
                embedding_index=embedding_index,
                json_record=record,
                crop_base_dir=crop_base_dir,
            )

            vehicle_count += 1

        return vehicle_count


# ── Pure functions (no service deps) ──────────────────────────────

def _compute_velocity(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Add image-space velocity columns to the tracking DataFrame.

    Added columns:
        center_x, center_y
        dx_pixels, dy_pixels
        dt_seconds
        vx_pixels_per_sec, vy_pixels_per_sec
        speed_pixels_per_sec

    The first detection of each track has NaN velocity (no prior frame).
    """
    df = df.copy()
    df["center_x"] = (df["x1"] + df["x2"]) / 2.0
    df["center_y"] = (df["y1"] + df["y2"]) / 2.0

    df = df.sort_values(["track_id", "frame_number"]).reset_index(drop=True)

    df["dx_pixels"] = df.groupby("track_id")["center_x"].diff()
    df["dy_pixels"] = df.groupby("track_id")["center_y"].diff()
    df["dt_seconds"] = df.groupby("track_id")["frame_number"].diff() / fps

    df["vx_pixels_per_sec"] = df["dx_pixels"] / df["dt_seconds"]
    df["vy_pixels_per_sec"] = df["dy_pixels"] / df["dt_seconds"]
    df["speed_pixels_per_sec"] = np.sqrt(
        df["vx_pixels_per_sec"] ** 2 + df["vy_pixels_per_sec"] ** 2
    )

    # Replace inf values with NaN
    inf_cols = [
        "dx_pixels", "dy_pixels", "dt_seconds",
        "vx_pixels_per_sec", "vy_pixels_per_sec", "speed_pixels_per_sec",
    ]
    df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
    return df


def _extract_crops(
    video_path: Path,
    tracks_df: pd.DataFrame,
    crops_root: Path,
    max_crops: int = 8,
) -> dict:
    """
    Fast, single-pass sequential crop extraction.
    Reads the video sequentially once instead of seeking hundreds of times.

    Returns
    -------
    dict mapping track_id (int) → list of {"frame_number": int, "crop_path": Path, "weight": float}
    """
    crops_root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video for crop extraction: %s", video_path)
        return {}

    # 1. Pre-plan which crops to extract per frame
    crops_to_extract: dict[int, list] = {}
    track_crops_map: dict[int, list] = {int(t): [] for t in tracks_df["track_id"].unique()}

    for track_id, group in tracks_df.groupby("track_id"):
        group = group.sort_values("frame_number").reset_index(drop=True)
        count = len(group)
        n_samples = min(max_crops, count)
        sample_idxs = np.linspace(0, count - 1, n_samples).round().astype(int)
        selected = group.iloc[sample_idxs]

        track_dir = crops_root / f"track_{int(track_id):05d}"
        track_dir.mkdir(parents=True, exist_ok=True)

        for _, row in selected.iterrows():
            frame_no = int(row["frame_number"])
            crop_path = track_dir / f"frame_{frame_no:06d}.jpg"
            conf = float(row.get("confidence", 0.5))
            crops_to_extract.setdefault(frame_no, []).append(
                (int(track_id), int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]), crop_path, conf)
            )

    # 2. Single sequential linear scan over video
    curr_frame = 0
    try:
        while crops_to_extract:
            success, frame = cap.read()
            if not success:
                break
            curr_frame += 1

            if curr_frame in crops_to_extract:
                targets = crops_to_extract.pop(curr_frame)
                h, w = frame.shape[:2]
                for tid, x1, y1, x2, y2, crop_path, conf in targets:
                    cx1, cy1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                    cx2, cy2 = max(0, min(w, x2)), max(0, min(h, y2))
                    if cx2 > cx1 and cy2 > cy1:
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            cv2.imwrite(str(crop_path), crop)
                            area = (cx2 - cx1) * (cy2 - cy1)
                            weight = conf * float(np.sqrt(max(1, area)))
                            track_crops_map[tid].append(
                                {
                                    "frame_number": curr_frame,
                                    "crop_path": crop_path,
                                    "weight": weight,
                                }
                            )
    finally:
        cap.release()

    return track_crops_map
