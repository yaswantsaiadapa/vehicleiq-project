"""
VehicleIQ — AttributeService
==============================
Zero-shot vehicle attribute classification using SigLIP2.

The attribute prompts are pre-encoded once at startup.  For each new
vehicle embedding, classification is a fast dot-product against the
cached prompt embeddings.

Returned scores are SigLIP cosine similarities in [-1, 1].
They are NOT calibrated probabilities.
"""

import logging

import numpy as np

from app import config
from app.services.embedding import EmbeddingService

log = logging.getLogger(__name__)


class AttributeService:
    """
    Zero-shot attribute classifier using SigLIP2 similarity.

    Attributes classified:
        color        — 9 colour options
        body_style   — 8 body-style options
        orientation  — 8 orientation options
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._emb = embedding_service
        self._prompt_embeddings: dict[str, np.ndarray] = {}

    # ── Initialisation ────────────────────────────────────────────

    def _ensure_prompts(self) -> None:
        if self._prompt_embeddings:
            return
        log.info("Pre-encoding attribute prompts…")
        for attribute, prompts in config.ATTRIBUTE_PROMPTS.items():
            self._prompt_embeddings[attribute] = self._emb.encode_text(prompts)
            log.debug(
                "  %s: %d prompts → shape %s",
                attribute,
                len(prompts),
                self._prompt_embeddings[attribute].shape,
            )
        log.info("Attribute prompts ready")

    # ── Classification ────────────────────────────────────────────

    def classify(self, vehicle_embedding: np.ndarray) -> dict:
        """
        Classify a single vehicle embedding across all attribute groups.

        Parameters
        ----------
        vehicle_embedding:
            L2-normalised numpy array of shape (768,).

        Returns
        -------
        dict mapping attribute name → {value: str, score: float}

        The score is the SigLIP cosine similarity (not a probability).
        """
        self._ensure_prompts()

        vec = vehicle_embedding.reshape(1, -1).astype(np.float32)  # (1, 768)

        result: dict = {}

        for attribute, prompt_embeddings in self._prompt_embeddings.items():
            # Dot product → cosine similarity (vectors are L2-normalised)
            similarities = vec @ prompt_embeddings.T  # (1, K)
            similarities = similarities.squeeze(0)  # (K,)

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_label = config.ATTRIBUTE_LABELS[attribute][best_idx]

            result[attribute] = {
                "value": best_label,
                "score": round(best_score, 6),
            }

        return result
