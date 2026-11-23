"""
VehicleIQ — /vehicles router
==============================
GET /vehicles/{vehicle_id}           — Full vehicle record + crop URLs
GET /videos/{video_id}/vehicles      — All vehicles for one video
GET /crops/{vehicle_id}/{path:path}  — Serve a crop image file
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["vehicles"])


@router.get("/vehicles/{vehicle_id}")
async def get_vehicle(vehicle_id: str, request: Request):
    """
    Return the full vehicle record including all crop image URLs.

    Attribute scores are SigLIP cosine similarities — NOT probabilities.
    """
    db = request.app.state.db
    vehicle = db.get_vehicle(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle_id}' not found")

    record = vehicle.get("json_record") or {}
    base_url = str(request.base_url).rstrip("/")

    # Build full crop URLs
    crops_relative = record.get("visual", {}).get("representative_crops", [])
    crop_urls = [
        f"{base_url}/crops/{vehicle_id}/{c}" for c in crops_relative
    ]

    return {
        "vehicle_id": vehicle_id,
        "video_id": vehicle.get("video_id"),
        "track_id": vehicle.get("track_id"),
        "tracking": record.get("tracking", {}),
        "vehicle": record.get("vehicle", {}),
        "velocity": record.get("velocity", {}),
        "crop_urls": crop_urls,
        "db": {
            "vehicle_type": vehicle.get("vehicle_type"),
            "vehicle_type_score": vehicle.get("vehicle_type_confidence"),
            "color": vehicle.get("color"),
            "color_score": vehicle.get("color_confidence"),
            "body_style": vehicle.get("body_style"),
            "body_style_score": vehicle.get("body_style_confidence"),
            "orientation": vehicle.get("orientation"),
            "orientation_score": vehicle.get("orientation_confidence"),
            "mean_speed": vehicle.get("mean_speed"),
            "median_speed": vehicle.get("median_speed"),
            "max_speed": vehicle.get("max_speed"),
        },
    }


@router.get("/videos/{video_id}/vehicles")
async def list_vehicles_for_video(
    video_id: str,
    request: Request,
    limit: int = 100,
    offset: int = 0,
):
    """Return all vehicles detected in a given video."""
    db = request.app.state.db

    video = db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    vehicles = db.get_vehicles_for_video(video_id)
    total = len(vehicles)
    page = vehicles[offset : offset + limit]

    base_url = str(request.base_url).rstrip("/")
    result = []
    for v in page:
        record = v.get("json_record") or {}
        crops = record.get("visual", {}).get("representative_crops", [])
        thumbnail = (
            f"{base_url}/crops/{v['vehicle_id']}/{crops[0]}"
            if crops
            else None
        )
        result.append(
            {
                "vehicle_id": v["vehicle_id"],
                "track_id": v["track_id"],
                "vehicle_type": v["vehicle_type"],
                "color": v["color"],
                "body_style": v["body_style"],
                "first_frame": v["first_frame"],
                "last_frame": v["last_frame"],
                "thumbnail_url": thumbnail,
            }
        )

    return {
        "video_id": video_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "vehicles": result,
    }


@router.get("/crops/{vehicle_id}/{path:path}")
async def serve_crop(vehicle_id: str, path: str, request: Request):
    """
    Serve a crop image for a vehicle.

    Resolves the full file path from the vehicle's crop_base_dir in the
    database and returns the image file directly.
    """
    db = request.app.state.db
    vehicle = db.get_vehicle(vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle_id}' not found")

    crop_base = vehicle.get("crop_base_dir")
    if not crop_base:
        raise HTTPException(status_code=404, detail="No crop directory for this vehicle")

    full_path = Path(crop_base) / path
    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Crop image not found: {path}",
        )

    # Basic security: ensure the resolved path stays inside crop_base
    try:
        full_path.resolve().relative_to(Path(crop_base).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal detected")

    return FileResponse(str(full_path), media_type="image/jpeg")
