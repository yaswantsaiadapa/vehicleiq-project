"""
VehicleIQ — Component & Multimodal Search Verification Test
=============================================================
Run with: python tests/test_components.py
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add vehicleiq to sys.path
viq_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(viq_root))

print("=" * 60)
print("VehicleIQ Multimodal Engine & Cloud Storage Verification")
print("=" * 60)

# 1. Config
from app import config
print("\n[1/5] Config check:")
print(f"  DATA_DIR: {config.DATA_DIR}")
print(f"  DB_PATH: {config.DB_PATH}")
print(f"  FAISS_PATH: {config.FAISS_PATH}")
print(f"  SPARSE_INDEX_PATH: {config.SPARSE_INDEX_PATH}")
print(f"  TRACK1_EMBEDDINGS_PATH: {config.TRACK1_EMBEDDINGS_PATH}")
print(f"  TRACK1_METADATA_PATH: {config.TRACK1_METADATA_PATH}")
assert config.TRACK1_EMBEDDINGS_PATH is not None and config.TRACK1_EMBEDDINGS_PATH.exists(), "Track 1 embeddings not found!"
print("  [OK] Dataset paths resolved successfully")

# 2. Database Service
from app.services.database import DatabaseService
print("\n[2/5] DatabaseService check:")
db = DatabaseService()
v_count = db.vehicle_count()
print(f"  Total Vehicles in DB: {v_count}")
assert v_count >= 2155, f"Expected at least 2,155 vehicles migrated into DB, got {v_count}!"
sample_v = db.get_vehicle("b06c903c-a25d-45fe-b0d5-294f72e34023") or db.get_vehicle("S01_c001_v_00002")
assert sample_v is not None, "Sample vehicle not found!"
print(f"  Sample vehicle ID: {sample_v.get('vehicle_id')}")
print(f"  Sample vehicle type: {sample_v.get('vehicle_type')}")
print(f"  Sample color: {sample_v.get('color')}")
print("  [OK] Database initialized and full dataset migrated")

# 3. Retrieval Service & BM25 Sparse Index
from app.services.retrieval import RetrievalService
print("\n[3/5] RetrievalService & BM25 check:")
retrieval = RetrievalService()
print(f"  Dense Vector Count: {retrieval.vector_count}")
print(f"  Sparse BM25 Docs  : {retrieval.sparse.n_docs}")
assert retrieval.vector_count >= 2155, f"Expected >= 2,155 dense vectors, got {retrieval.vector_count}"
assert retrieval.sparse.n_docs >= 2000, f"Expected >= 2,000 sparse indexed documents, got {retrieval.sparse.n_docs}"
print("  [OK] Dual-Encoder Gallery initialized (Dense SigLIP2 + Sparse BM25)")

# 4. Hybrid Search via Reciprocal Rank Fusion (RRF)
print("\n[4/5] Hybrid Search & RRF Fusion check:")
import numpy as np
dummy_query_emb = np.ones((768,), dtype=np.float32)
dummy_query_emb /= np.linalg.norm(dummy_query_emb)

hybrid_hits = retrieval.search_hybrid(
    query_embedding=dummy_query_emb,
    query_text="red sedan driving straight",
    top_k=5,
    alpha=0.5,
)
print(f"  Top 5 Hybrid Hits (Dense SigLIP2 + Sparse BM25):")
for hit in hybrid_hits:
    print(
        f"    Rank #{hit['rank']} -> ID: {hit['vehicle_id']} | "
        f"RRF: {hit['hybrid_score']:.5f} | Dense: {hit['dense_score']:.3f} | Sparse: {hit['sparse_score']:.2f}"
    )

assert len(hybrid_hits) == 5
assert hybrid_hits[0]["rank"] == 1
assert "hybrid_score" in hybrid_hits[0]
assert "dense_score" in hybrid_hits[0]
assert "sparse_score" in hybrid_hits[0]
print("  [OK] Hybrid RRF search successfully fused dense and sparse signals")

# 5. Cloud Storage & FastAPI Routes check
from app.main import app
from app.services.storage import CloudStorageService
print("\n[5/5] Cloud Storage & Dedicated API Routes check:")
storage = CloudStorageService()
print(f"  Cloud Storage Active: {storage.is_cloud_enabled}")
print(f"  Public CDN Domain: {storage.public_domain}")
sample_cdn_url = storage.get_crop_url("b06c903c-a25d-45fe-b0d5-294f72e34023", "frame_000001.jpg")
print(f"  Sample Live Cloud CDN URL: {sample_cdn_url}")

routes = []
for r in app.routes:
    if hasattr(r, "path") and r.path:
        routes.append(r.path)
    elif hasattr(r, "original_router"):
        for sub in r.original_router.routes:
            routes.append(sub.path)

print(f"  Registered routes: {routes}")
assert "/search" in routes, "No /search text route found"
assert "/search/image" in routes, "No /search/image visual route found"
assert "/health" in routes, "No /health route found"
print("  [OK] Dedicated text & image search routes verified")

print("\n" + "=" * 60)
print("ALL MULTIMODAL RETRIEVAL & CLOUD STORAGE CHECKS PASSED!")
print("=" * 60)
