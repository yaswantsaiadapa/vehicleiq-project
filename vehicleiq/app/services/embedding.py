"""
VehicleIQ — EmbeddingService
==============================
Loads SigLIP2 (google/siglip2-base-patch16-224) once and exposes
methods for:

    encode_images  — per-image embeddings (N, 768)
    encode_text    — per-text embeddings (N, 768)
    encode_vehicle — mean-pooled vehicle embedding (768,)

All returned embeddings are L2-normalised float32 numpy arrays.
GPU (float16) is used when available; CPU (float32) otherwise.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app import config

log = logging.getLogger(__name__)


class EmbeddingService:
    """
    SigLIP2 encoder service.

    Models are loaded lazily on first inference call so application startup
    is instantaneous and does not block the web server.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = None
        self.model = None
        self._is_loaded = False

    def _ensure_loaded(self) -> None:
        if self._is_loaded and self.model is not None and self.processor is not None:
            return

        from transformers import AutoModel, AutoProcessor

        local_model_dir = Path(config.DATA_DIR) / "models" / "siglip2"
        if (local_model_dir / "model.safetensors").exists() and (local_model_dir / "model.safetensors").stat().st_size > 500_000_000:
            model_target = str(local_model_dir.resolve())
            log.info("Loading SigLIP2 from local storage: %s", model_target)
        else:
            model_target = config.SIGLIP2_MODEL_ID
            log.info(
                "Loading SigLIP2: %s  (device=%s  dtype=%s)",
                model_target,
                self.device,
                self.dtype,
            )

        try:
            self.processor = AutoProcessor.from_pretrained(model_target, use_fast=True)
        except Exception:
            try:
                self.processor = AutoProcessor.from_pretrained(model_target, use_fast=False)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(model_target, local_files_only=True)

        try:
            self.model = AutoModel.from_pretrained(
                model_target, dtype=self.dtype
            ).to(self.device)
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_target, dtype=self.dtype, local_files_only=True
            ).to(self.device)

        self.model.eval()
        self._is_loaded = True
        log.info("SigLIP2 ready")

    # ── Image encoding ────────────────────────────────────────────

    def encode_images(
        self, images: list[Image.Image], batch_size: int | None = None
    ) -> np.ndarray:
        """
        Encode a list of PIL images.

        Returns
        -------
        np.ndarray  shape (N, 768) float32, L2-normalised
        """
        if not images:
            return np.empty((0, 768), dtype=np.float32)

        self._ensure_loaded()

        bs = batch_size or config.EMBEDDING_BATCH_SIZE
        all_embeddings: list[torch.Tensor] = []

        for start in range(0, len(images), bs):
            batch = images[start : start + bs]

            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if torch.is_tensor(v)
            }

            with torch.inference_mode():
                output = self.model.get_image_features(**inputs)

            embeddings = (
                output.pooler_output if hasattr(output, "pooler_output") else output
            )
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy().astype(np.float32)

    # ── Text encoding ─────────────────────────────────────────────

    def encode_text(
        self, texts: list[str] | str, batch_size: int | None = None
    ) -> np.ndarray:
        """
        Encode a list of text strings.

        Returns
        -------
        np.ndarray  shape (N, 768) float32, L2-normalised
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        self._ensure_loaded()

        bs = batch_size or config.EMBEDDING_BATCH_SIZE
        all_embeddings: list[torch.Tensor] = []

        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]

            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if torch.is_tensor(v)
            }

            with torch.inference_mode():
                output = self.model.get_text_features(**inputs)

            embeddings = (
                output.pooler_output if hasattr(output, "pooler_output") else output
            )
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy().astype(np.float32)

    # ── Vehicle embedding ─────────────────────────────────────────

    def encode_vehicle(
        self,
        images: list[Image.Image],
        weights: list[float] | np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Quality-weighted temporal crop embeddings → single L2-normalised
        vehicle vector (768,).

        Parameters
        ----------
        images:
            List of PIL images representing the vehicle.
        weights:
            Optional weights for each image (e.g. confidence * sqrt(area)).
            If None, uniform mean-pooling is applied.

        Returns None if no valid images are provided.
        """
        if not images:
            return None

        frame_embeddings = self.encode_images(images)  # (N, 768)
        if frame_embeddings.shape[0] == 0:
            return None

        if weights is not None and len(weights) == frame_embeddings.shape[0]:
            w = np.array(weights, dtype=np.float32)
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
                vehicle_vec = np.sum(frame_embeddings * w[:, None], axis=0)
            else:
                vehicle_vec = frame_embeddings.mean(axis=0)
        else:
            vehicle_vec = frame_embeddings.mean(axis=0)

        norm = np.linalg.norm(vehicle_vec)
        if norm > 0:
            vehicle_vec = vehicle_vec / norm

        return vehicle_vec.astype(np.float32)

    def encode_vehicle_from_paths(
        self, crop_paths: list[Path]
    ) -> np.ndarray | None:
        """Convenience wrapper: load PIL Images then encode."""
        images: list[Image.Image] = []
        for p in crop_paths:
            if p.exists():
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    log.warning("Could not open crop %s: %s", p, e)
        return self.encode_vehicle(images)
