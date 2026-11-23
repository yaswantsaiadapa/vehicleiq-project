"""
VehicleIQ — FastAPI Application
=================================
Dedicated Multimodal Vehicle Retrieval Engine.

Features:
  - Dual-mode Search:
      * Natural Language Text Search (Dense SigLIP2 + Sparse BM25 + Reciprocal Rank Fusion)
      * Visual Re-ID Image Search (SigLIP2 Vision Encoder + Cosine Similarity)
  - Decoupled Cloud Media Storage (Supabase Storage / AWS S3 CDN streaming)
  - Sub-15ms fast vector search across 2,320 indexed vehicles

API surface
-----------
POST   /search                           natural-language hybrid search (RRF)
POST   /search/image                     visual image re-identification search
GET    /vehicles/{id}                    full vehicle record & metadata
GET    /health                           service status
"""

import logging
import sys
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import config
from app.routers import search, vehicles
from app.services.database import DatabaseService
from app.services.embedding import EmbeddingService
from app.services.retrieval import RetrievalService
from app.services.storage import CloudStorageService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load database, embedding models, vector index, and cloud storage.
    Shutdown: persist index state.
    """
    log.info("=" * 60)
    log.info("  VehicleIQ — Multimodal Retrieval Engine starting up")
    log.info("=" * 60)

    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.info("Running on CPU. Fast vector inference ready.")

    config.ensure_dirs()

    log.info("Loading core services…")
    app.state.db = DatabaseService()
    app.state.embedding = EmbeddingService()
    app.state.retrieval = RetrievalService()
    app.state.storage = CloudStorageService()

    log.info(
        "All services ready — DB vehicles: %d | Vectors: %d | Cloud Storage: %s",
        app.state.db.vehicle_count(),
        app.state.retrieval.vector_count,
        "Active (Supabase)" if app.state.storage.is_cloud_enabled else "Local",
    )
    log.info("=" * 60)

    yield  # ── app is serving ──────────────────────────────────

    log.info("Shutting down…")
    app.state.retrieval.save()
    log.info("Shutdown complete")


# ── Application ───────────────────────────────────────────────────

app = FastAPI(
    title="VehicleIQ",
    description=(
        "Multimodal Vehicle Retrieval Engine: search vehicles by natural-language "
        "description or uploaded image using SigLIP2 + BM25 + Cloud CDN."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers (Dedicated Retrieval & Vehicle Metadata)
app.include_router(search.router)
app.include_router(vehicles.router)


# ── Health ────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health(request: Request):
    """Return service readiness status."""
    state = request.app.state
    return JSONResponse(
        {
            "status": "ok",
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "siglip_ready": state.embedding is not None,
            "faiss_vectors": state.retrieval.vector_count,
            "db_vehicles": state.db.vehicle_count(),
            "cloud_storage_enabled": state.storage.is_cloud_enabled,
        }
    )
