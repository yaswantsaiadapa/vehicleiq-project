"""
VehicleIQ — RetrievalService
==============================
Manages the multimodal hybrid search gallery for semantic vehicle retrieval.

Architecture:
  - Dense Retriever : SigLIP2 (768-D) via FAISS IndexFlatIP (NumPy dot-product fallback)
  - Sparse Retriever: BM25Okapi lexical inverted index over vehicle descriptions
  - Hybrid Fusion   : Reciprocal Rank Fusion (RRF) combining dense & sparse ranks

On startup, it automatically performs migration to ensure the full CityFlow-NL
gallery (2,155 vehicles) plus any prototype surveillance videos (165 vehicles)
are indexed, yielding 2,320 total searchable vehicles.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app import config
from app.services.sparse import BM25Index

log = logging.getLogger(__name__)


class RetrievalService:
    """
    Dual-Encoder Hybrid Retrieval Engine.
    Combines dense SigLIP2 cosine similarity with sparse BM25 lexical ranking
    via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            log.warning(
                "faiss is not installed in this environment; using NumPy "
                "cosine similarity fallback (fast & zero external deps)."
            )

        self.index = None
        self._embeddings: np.ndarray = np.empty((0, 768), dtype=np.float32)
        self.vehicle_ids: list[str] = []
        self.sparse = BM25Index()

        config.FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._npy_path = config.FAISS_PATH.with_name("vehicle_embeddings.npy")
        self._load_or_migrate()

    # ── Initialisation & Migration ─────────────────────────────────

    def _load_or_migrate(self) -> None:
        if self._has_saved_index():
            self._load()
            # If the saved index only has the 165 prototype vehicles, upgrade to full dataset
            if self.vector_count < 1000 and self._track1_available():
                log.info("Detected legacy 165-vehicle index. Upgrading to full 2,155+ dataset...")
                self._migrate_full_dataset()
        elif self._track1_available() or self._track2_available():
            log.info("Running initial hybrid dataset migration into vector & lexical index...")
            self._migrate_full_dataset()
        else:
            log.info("Starting with an empty vector and lexical index (dim=768)")
            if self._faiss is not None:
                self.index = self._faiss.IndexFlatIP(768)
            self._embeddings = np.empty((0, 768), dtype=np.float32)
            self.vehicle_ids = []

    def _has_saved_index(self) -> bool:
        if not config.FAISS_IDS_PATH.exists():
            return False
        if self._faiss is not None and config.FAISS_PATH.exists():
            return True
        return self._npy_path.exists()

    def _track1_available(self) -> bool:
        return (
            config.TRACK1_EMBEDDINGS_PATH is not None
            and config.TRACK1_EMBEDDINGS_PATH.exists()
            and config.TRACK1_IDS_PATH is not None
            and config.TRACK1_IDS_PATH.exists()
        )

    def _track2_available(self) -> bool:
        return (
            config.TRACK2_EMBEDDINGS_PATH is not None
            and config.TRACK2_EMBEDDINGS_PATH.exists()
            and config.TRACK2_IDS_PATH is not None
            and config.TRACK2_IDS_PATH.exists()
        )

    def _load(self) -> None:
        log.info("Loading vector gallery from %s", config.FAISS_PATH.parent)
        with open(config.FAISS_IDS_PATH, encoding="utf-8") as f:
            self.vehicle_ids = json.load(f)

        if self._faiss is not None and config.FAISS_PATH.exists():
            self.index = self._faiss.read_index(str(config.FAISS_PATH))
            if self._npy_path.exists():
                try:
                    self._embeddings = np.load(str(self._npy_path)).astype(np.float32)
                except Exception:
                    pass
        elif self._npy_path.exists():
            self._embeddings = np.load(str(self._npy_path)).astype(np.float32)

        # Load BM25 index
        if config.SPARSE_INDEX_PATH.exists():
            self.sparse.load(config.SPARSE_INDEX_PATH)
        else:
            log.info("Sparse BM25 index not found on disk; building from available metadata...")
            self._rebuild_sparse_index()

        log.info(
            "Vector & lexical gallery ready — %d dense vectors, %d sparse docs",
            self.vector_count,
            self.sparse.n_docs,
        )

    def _migrate_full_dataset(self) -> None:
        """
        Build the unified gallery combining Track 1 (2,155 vehicles) and
        Track 2 (165 prototype surveillance vehicles).
        Total gallery: ~2,320 vehicles.
        """
        all_embeddings: list[np.ndarray] = []
        all_ids: list[str] = []

        # 1. Load Track 1 (CityFlow-NL Benchmark: 2,155 vehicles)
        if self._track1_available():
            log.info("Loading Track 1 benchmark embeddings (2,155 vehicles)...")
            t1_embs = np.load(str(config.TRACK1_EMBEDDINGS_PATH)).astype(np.float32)
            with open(config.TRACK1_IDS_PATH, encoding="utf-8") as f:
                t1_ids = json.load(f)
            all_embeddings.append(t1_embs)
            all_ids.extend([str(vid) for vid in t1_ids])
            log.info("Track 1 loaded: %d embeddings", len(t1_ids))

        # 2. Load Track 2 (Prototype Surveillance Video: 165 vehicles)
        if self._track2_available():
            log.info("Loading Track 2 surveillance video embeddings (165 vehicles)...")
            t2_embs = np.load(str(config.TRACK2_EMBEDDINGS_PATH)).astype(np.float32)
            with open(config.TRACK2_IDS_PATH, encoding="utf-8") as f:
                t2_raw_ids = json.load(f)
            t2_ids = [
                f"S01_c001_v_{tid:05d}" if isinstance(tid, int) else str(tid)
                for tid in t2_raw_ids
            ]
            all_embeddings.append(t2_embs)
            all_ids.extend(t2_ids)
            log.info("Track 2 loaded: %d embeddings", len(t2_ids))

        if not all_ids:
            log.warning("No dataset embeddings available for migration!")
            return

        combined_embeddings = np.vstack(all_embeddings).astype(np.float32)
        dim = combined_embeddings.shape[1]

        self._embeddings = combined_embeddings
        self.vehicle_ids = all_ids

        if self._faiss is not None:
            self.index = self._faiss.IndexFlatIP(dim)
            self.index.add(combined_embeddings)

        # 3. Build the BM25 Sparse Index
        self._rebuild_sparse_index()

        # 4. Save unified indices to disk
        self.save()
        log.info(
            "Unified gallery migration complete — %d total vehicles indexed!",
            len(all_ids),
        )

    def _rebuild_sparse_index(self) -> None:
        """Construct BM25 index from vehicle text descriptions and attributes."""
        corpus: dict[str, str] = {}

        # 1. Index Track 1 descriptions
        if config.TRACK1_METADATA_PATH and config.TRACK1_METADATA_PATH.exists():
            try:
                with open(config.TRACK1_METADATA_PATH, "r", encoding="utf-8") as f:
                    t1_meta = json.load(f)
                for vid, rec in t1_meta.items():
                    desc_list = rec.get("descriptions", [])
                    other_views = rec.get("descriptions_other_views", [])
                    all_text = " ".join(desc_list + other_views)
                    corpus[str(vid)] = all_text
            except Exception as e:
                log.warning("Could not read Track 1 metadata for BM25: %s", e)

        # 2. Index Track 2 attributes
        if config.TRACK2_DB_PATH and config.TRACK2_DB_PATH.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(config.TRACK2_DB_PATH))
                conn.row_factory = sqlite3.Row
                for row in conn.execute("SELECT * FROM vehicles"):
                    d = dict(row)
                    vid = d.get("vehicle_id")
                    if vid:
                        attrs = f"{d.get('color', '')} {d.get('vehicle_type', '')} {d.get('body_style', '')} {d.get('orientation', '')}"
                        corpus[str(vid)] = attrs
                conn.close()
            except Exception as e:
                log.warning("Could not read Track 2 attributes for BM25: %s", e)

        if corpus:
            self.sparse.fit(corpus)
            self.sparse.save(config.SPARSE_INDEX_PATH)

    # ── Dense Search ──────────────────────────────────────────────

    def search_dense(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Dense semantic search using SigLIP2 cosine similarity."""
        n_total = self.vector_count
        if n_total == 0:
            return []

        k = min(top_k, n_total)
        vec = query_embedding.reshape(1, -1).astype(np.float32)

        if self.index is not None:
            scores, indices = self.index.search(vec, k)
            scores_list = scores[0].tolist()
            indices_list = indices[0].tolist()
        else:
            sims = (vec @ self._embeddings.T).squeeze(0)
            top_indices = np.argsort(-sims)[:k]
            indices_list = top_indices.tolist()
            scores_list = [float(sims[idx]) for idx in indices_list]

        results = []
        for rank, (score, idx) in enumerate(zip(scores_list, indices_list), start=1):
            if idx < 0 or idx >= len(self.vehicle_ids):
                continue
            results.append(
                {
                    "rank": rank,
                    "vehicle_id": self.vehicle_ids[idx],
                    "score": round(float(score), 6),
                }
            )
        return results

    # ── Sparse Search ─────────────────────────────────────────────

    def search_sparse(self, query_text: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Sparse lexical search using BM25Okapi."""
        return self.sparse.search(query_text, top_k=top_k)

    # ── Hybrid Retrieval (Dense + Sparse + RRF) ───────────────────

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Multimodal Hybrid Search via Reciprocal Rank Fusion (RRF).

        Parameters
        ----------
        query_embedding:
            SigLIP2 768-D query vector.
        query_text:
            Raw natural-language search string.
        top_k:
            Number of ranked results to return.
        alpha:
            Weight between dense and sparse [0, 1]. Default 0.5 (balanced).
            Higher alpha favors visual semantics; lower alpha favors exact keywords.
        rrf_k:
            RRF smoothing factor (standard: 60).

        Returns
        -------
        list of dicts with hybrid_score, dense_score, sparse_score, and ranks.
        """
        candidate_pool = max(top_k * 3, 50)

        # 1. Retrieve candidates from both modalities
        dense_hits = self.search_dense(query_embedding, top_k=candidate_pool)
        sparse_hits = self.search_sparse(query_text, top_k=candidate_pool)

        dense_map = {h["vehicle_id"]: h for h in dense_hits}
        sparse_map = {h["vehicle_id"]: h for h in sparse_hits}

        all_candidates = set(dense_map.keys()) | set(sparse_map.keys())
        if not all_candidates:
            return []

        default_rank = candidate_pool + 10
        fused_results = []

        for vid in all_candidates:
            d_hit = dense_map.get(vid)
            s_hit = sparse_map.get(vid)

            d_rank = d_hit["rank"] if d_hit else default_rank
            d_score = d_hit["score"] if d_hit else -1.0

            s_rank = s_hit["rank"] if s_hit else default_rank
            s_score = s_hit["score"] if s_hit else 0.0

            # Reciprocal Rank Fusion formula
            rrf_score = (alpha / (d_rank + rrf_k)) + ((1.0 - alpha) / (s_rank + rrf_k))

            fused_results.append(
                {
                    "vehicle_id": vid,
                    "hybrid_score": round(rrf_score, 6),
                    "dense_score": round(d_score, 4),
                    "sparse_score": round(s_score, 4),
                    "dense_rank": d_rank if d_hit else None,
                    "sparse_rank": s_rank if s_hit else None,
                }
            )

        # Sort by hybrid RRF score descending
        fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = fused_results[:top_k]

        for rank_idx, item in enumerate(top_results, start=1):
            item["rank"] = rank_idx

        return top_results

    # ── Backward-compatible search method ──────────────────────────

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        """Dense search for backward compatibility."""
        return self.search_dense(query_embedding, top_k=top_k)

    # ── Mutation & Persistence ────────────────────────────────────

    def add_vehicle(
        self,
        vehicle_id: str,
        embedding: np.ndarray,
        text_description: str | None = None,
    ) -> int:
        """Add vehicle to both dense and sparse indices."""
        vec = embedding.reshape(1, -1).astype(np.float32)
        with self._lock:
            if self.index is not None:
                self.index.add(vec)

            if len(self._embeddings) == 0:
                self._embeddings = vec
            else:
                self._embeddings = np.vstack([self._embeddings, vec])

            embedding_index = len(self.vehicle_ids)
            self.vehicle_ids.append(vehicle_id)

            if text_description:
                self.sparse.add_document(vehicle_id, text_description)

        return embedding_index

    def save(self) -> None:
        """Persist dense FAISS index, npy array, vehicle_ids, and BM25 index."""
        with self._lock:
            if self.index is not None and self._faiss is not None:
                self._faiss.write_index(self.index, str(config.FAISS_PATH))

            if len(self._embeddings) > 0:
                np.save(str(self._npy_path), self._embeddings)

            with open(config.FAISS_IDS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.vehicle_ids, f, indent=2)

            self.sparse.save(config.SPARSE_INDEX_PATH)

        log.info(
            "Gallery saved: %d dense vectors, %d sparse docs",
            self.vector_count,
            self.sparse.n_docs,
        )

    @property
    def vector_count(self) -> int:
        if self.index is not None:
            return self.index.ntotal
        return len(self.vehicle_ids)
