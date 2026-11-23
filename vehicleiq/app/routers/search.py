"""
VehicleIQ — /search router
============================
Supports dual-mode multimodal search:
  1. POST /search       — Text Query: Dual-Encoder Hybrid Search (Dense SigLIP2 + Sparse BM25 + RRF)
  2. POST /search/image — Image Query: Visual Re-ID Search (SigLIP2 Vision Encoder + Cosine Similarity)

Both modes fetch enriched metadata from SQLite and stream thumbnails via Supabase CDN.
"""

import io
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512, description="Natural-language vehicle description")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid balance weight: 0.0 = pure sparse BM25, 1.0 = pure dense SigLIP2, 0.5 = balanced RRF",
    )


def _first_crop_url(vehicle_id: str, vehicle: dict, request: Request) -> str | None:
    """Return the URL for the first available crop image (Cloud CDN or local server)."""
    record = vehicle.get("json_record") or {}
    crops = record.get("visual", {}).get("representative_crops", [])

    first_file = crops[0] if crops else "frame_000001.jpg"
    # If crops has full relative path, take the filename
    if "/" in first_file or "\\" in first_file:
        first_file = first_file.replace("\\", "/").split("/")[-1]

    # Check if cloud storage is configured
    storage = getattr(request.app.state, "storage", None)
    if storage and storage.is_cloud_enabled:
        return storage.get_crop_url(vehicle_id, first_file)

    if not crops:
        return None

    crop_path = crops[0]
    if crop_path.startswith("http://") or crop_path.startswith("https://"):
        return crop_path

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/crops/{vehicle_id}/{crop_path}"


def _enrich_results(hits: list[dict[str, Any]], db, request: Request) -> list[dict[str, Any]]:
    """Enrich retrieval hit list with vehicle attributes, descriptions, and CDN thumbnail URLs."""
    vehicle_ids = [h["vehicle_id"] for h in hits]
    vehicles_by_id = {v["vehicle_id"]: v for v in db.get_vehicles_by_ids(vehicle_ids)}

    results: list[dict[str, Any]] = []
    for hit in hits:
        vid_id = hit["vehicle_id"]
        vehicle = vehicles_by_id.get(vid_id)
        if not vehicle:
            results.append(
                {
                    "vehicle_id": vid_id,
                    "rank": hit["rank"],
                    "score": hit.get("dense_score") or hit.get("score", 0.0),
                    "dense_score": hit.get("dense_score") or hit.get("score", 0.0),
                    "sparse_score": hit.get("sparse_score", 0.0),
                    "hybrid_score": hit.get("hybrid_score") or hit.get("score", 0.0),
                    "dense_rank": hit.get("dense_rank"),
                    "sparse_rank": hit.get("sparse_rank"),
                    "video_id": "cityflow_benchmark",
                    "vehicle_type": "vehicle",
                    "color": "unknown",
                    "body_style": "vehicle",
                    "orientation": "unspecified",
                    "track_id": 0,
                    "first_frame": 1,
                    "last_frame": 1,
                    "mean_speed": None,
                    "thumbnail_url": None,
                }
            )
            continue

        crop_url = _first_crop_url(vid_id, vehicle, request)
        rec = vehicle.get("json_record") or {}
        descriptions = rec.get("descriptions", [])

        results.append(
            {
                "vehicle_id": vid_id,
                "rank": hit["rank"],
                "score": hit.get("dense_score") or hit.get("score", 0.0),
                "dense_score": hit.get("dense_score") or hit.get("score", 0.0),
                "sparse_score": hit.get("sparse_score", 0.0),
                "hybrid_score": hit.get("hybrid_score") or hit.get("score", 0.0),
                "dense_rank": hit.get("dense_rank"),
                "sparse_rank": hit.get("sparse_rank"),
                "video_id": vehicle.get("video_id"),
                "vehicle_type": vehicle.get("vehicle_type"),
                "color": vehicle.get("color"),
                "body_style": vehicle.get("body_style"),
                "orientation": vehicle.get("orientation"),
                "track_id": vehicle.get("track_id"),
                "first_frame": vehicle.get("first_frame"),
                "last_frame": vehicle.get("last_frame"),
                "mean_speed": vehicle.get("mean_speed"),
                "descriptions": descriptions[:2] if descriptions else None,
                "thumbnail_url": crop_url,
            }
        )
    return results


@router.post("")
def search_vehicles(body: SearchRequest, request: Request):
    """
    Multimodal Hybrid Vehicle Search by Natural Language Text.
    Fuses dense SigLIP2 visual semantics with sparse BM25 lexical precision using RRF.
    """
    emb_service = request.app.state.embedding
    retrieval = request.app.state.retrieval
    db = request.app.state.db

    if retrieval.vector_count == 0:
        raise HTTPException(
            status_code=503,
            detail="Search index is empty. Please verify that vehicle data is indexed.",
        )

    # 1. Encode query with SigLIP2 text encoder
    query_vec = emb_service.encode_text(body.query)
    if query_vec.shape[0] == 0:
        raise HTTPException(status_code=422, detail="Failed to encode query text")

    query_embedding = query_vec[0]

    # 2. Run Dual-Encoder Hybrid Search via RRF
    hits = retrieval.search_hybrid(
        query_embedding=query_embedding,
        query_text=body.query,
        top_k=body.top_k,
        alpha=body.alpha,
    )

    if not hits:
        return {"query": body.query, "results": [], "total": 0, "index_size": retrieval.vector_count}

    # 3. Enrich with database metadata & CDN URLs
    results = _enrich_results(hits, db, request)

    return {
        "query": body.query,
        "results": results,
        "total": len(results),
        "index_size": retrieval.vector_count,
        "sparse_docs": retrieval.sparse.n_docs,
        "retrieval_mode": "hybrid_rrf",
    }


@router.post("/image")
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(10),
):
    """
    Visual Re-Identification Search by Uploaded Vehicle Image.
    Encodes image with SigLIP2 vision encoder and runs dense vector retrieval (IndexFlatIP).
    """
    emb_service = request.app.state.embedding
    retrieval = request.app.state.retrieval
    db = request.app.state.db

    if retrieval.vector_count == 0:
        raise HTTPException(
            status_code=503,
            detail="Search index is empty. Please verify that vehicle data is indexed.",
        )

    # 1. Parse uploaded image
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. Encode image with SigLIP2 vision encoder
    query_vec = emb_service.encode_images([pil_image])
    if query_vec.shape[0] == 0:
        raise HTTPException(status_code=422, detail="Failed to encode query image")

    query_embedding = query_vec[0]

    # 3. Run dense vector search
    hits = retrieval.search_dense(query_embedding=query_embedding, top_k=top_k)

    if not hits:
        return {"filename": file.filename, "results": [], "total": 0, "index_size": retrieval.vector_count}

    # 4. Enrich with database metadata & CDN URLs
    results = _enrich_results(hits, db, request)

    return {
        "filename": file.filename,
        "results": results,
        "total": len(results),
        "index_size": retrieval.vector_count,
        "retrieval_mode": "siglip2_dense_visual",
    }
