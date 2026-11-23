"""
VehicleIQ — /videos router
============================
POST /videos      — Upload a video for processing (non-blocking)
GET  /videos      — List all videos
GET  /videos/{id} — Status and metadata for one video
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app import config

log = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])

_GPU_WARNING = (
    "No GPU detected. Video processing will run on CPU and may take "
    "15–20 minutes for a 2000-frame video. Processing has started in "
    "the background — you can poll this video_id for status updates."
)


@router.post("", status_code=202)
async def upload_video(file: UploadFile, request: Request):
    """
    Upload a video file for processing.

    The pipeline (YOLO → ByteTrack → embeddings → FAISS) runs in a
    background thread so this endpoint returns immediately with a
    video_id.  Poll GET /videos/{video_id} to track progress.

    Returns 202 Accepted with:
        video_id, status, gpu_warning (if no CUDA device)
    """
    import torch

    processor = request.app.state.processor
    db = request.app.state.db

    if not file.content_type or not file.content_type.startswith("video/"):
        # Accept common video MIME types and also application/octet-stream
        if file.content_type not in (
            "video/mp4", "video/avi", "video/x-msvideo", "video/quicktime",
            "video/x-matroska", "application/octet-stream",
        ):
            log.warning("Received content_type=%s — proceeding anyway", file.content_type)

    # Generate a unique video_id and save the file
    video_id = processor.generate_video_id()
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    video_path = config.VIDEOS_DIR / f"{video_id}{suffix}"

    log.info("Saving uploaded video: %s → %s", file.filename, video_path)

    with open(video_path, "wb") as f:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            f.write(chunk)

    # Get video metadata before processing
    try:
        info = processor._detector.get_video_info(video_path)
    except Exception:
        info = {"fps": 0, "frame_count": 0, "width": 0, "height": 0}

    crop_base_dir = str(config.CROPS_DIR / video_id)

    # Register in DB as 'queued'
    db.insert_video(
        video_id=video_id,
        original_filename=file.filename or "upload",
        fps=info["fps"],
        frame_count=info["frame_count"],
        width=info["width"],
        height=info["height"],
        crop_base_dir=crop_base_dir,
    )

    # Launch background processing (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        request.app.state.executor,
        processor.process_video,
        video_id,
        video_path,
    )

    response_data = {
        "video_id": video_id,
        "status": "queued",
        "filename": file.filename,
        "frame_count": info["frame_count"],
        "fps": info["fps"],
        "message": "Processing started in background. Poll GET /videos/{video_id} for status.",
    }

    if not torch.cuda.is_available():
        response_data["gpu_warning"] = _GPU_WARNING

    return JSONResponse(content=response_data, status_code=202)


@router.get("")
async def list_videos(request: Request):
    """Return all videos ordered by creation time (newest first)."""
    db = request.app.state.db
    videos = db.list_videos()
    return {"videos": videos, "total": len(videos)}


@router.get("/{video_id}")
async def get_video(video_id: str, request: Request):
    """
    Return processing status and metadata for a video.

    Status values:
        queued      — waiting for a processing slot
        processing  — pipeline is actively running
        complete    — all vehicles indexed
        failed      — pipeline error (see error_message)
    """
    db = request.app.state.db
    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
    return video
