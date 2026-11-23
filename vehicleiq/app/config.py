"""
VehicleIQ — Application Configuration
======================================
All configuration is read from environment variables (or a .env file
in the working directory).  No Colab or hardcoded paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (checks current dir and app root)
load_dotenv()
_app_env = Path(__file__).resolve().parent.parent / ".env"
if _app_env.exists():
    load_dotenv(_app_env)



def _path(env_key: str, default: str) -> Path:
    """Read an env var as a Path, resolve it, and ensure it exists."""
    raw = os.environ.get(env_key, default)
    return Path(raw)


def _str(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default)


def _float(env_key: str, default: float) -> float:
    return float(os.environ.get(env_key, str(default)))


def _int(env_key: str, default: int) -> int:
    return int(os.environ.get(env_key, str(default)))


# ── Storage roots ──────────────────────────────────────────────────
DATA_DIR: Path = _path("VEHICLEIQ_DATA_DIR", "./storage")
DB_PATH: Path = _path("VEHICLEIQ_DB_PATH", "./storage/db/vehicleiq.db")
FAISS_PATH: Path = _path("VEHICLEIQ_FAISS_PATH", "./storage/faiss/vehicle_gallery.faiss")
FAISS_IDS_PATH: Path = _path("VEHICLEIQ_FAISS_IDS_PATH", "./storage/faiss/vehicle_ids.json")
SPARSE_INDEX_PATH: Path = _path("VEHICLEIQ_SPARSE_PATH", "./storage/faiss/sparse_bm25_index.json")
VIDEOS_DIR: Path = _path("VEHICLEIQ_VIDEOS_DIR", "./storage/videos")
CROPS_DIR: Path = _path("VEHICLEIQ_CROPS_DIR", "./storage/crops")

# ── Models ────────────────────────────────────────────────────────
SIGLIP2_MODEL_ID: str = _str("SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-224")
YOLO_MODEL_NAME: str = _str("YOLO_MODEL_NAME", "yolo26n.pt")

# ── Detection ─────────────────────────────────────────────────────
YOLO_CONFIDENCE: float = _float("YOLO_CONFIDENCE", 0.25)
YOLO_IMGSZ: int = _int("YOLO_IMGSZ", 640)

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASSES: list[int] = list(VEHICLE_CLASS_IDS.keys())

# ── Pipeline ──────────────────────────────────────────────────────
MAX_TEMPORAL_CROPS: int = _int("MAX_TEMPORAL_CROPS", 8)
EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 32)

# ── Attribute classification prompts ─────────────────────────────
ATTRIBUTE_PROMPTS: dict[str, list[str]] = {
    "color": [
        "a red vehicle",
        "a blue vehicle",
        "a white vehicle",
        "a black vehicle",
        "a gray vehicle",
        "a silver vehicle",
        "a green vehicle",
        "a yellow vehicle",
        "a brown vehicle",
    ],
    "body_style": [
        "a sedan",
        "an SUV",
        "a hatchback",
        "a pickup truck",
        "a van",
        "a bus",
        "a motorcycle",
        "a truck",
    ],
    "orientation": [
        "a vehicle viewed from the front",
        "a vehicle viewed from the rear",
        "a vehicle viewed from the left side",
        "a vehicle viewed from the right side",
        "a vehicle viewed from the front-left",
        "a vehicle viewed from the front-right",
        "a vehicle viewed from the rear-left",
        "a vehicle viewed from the rear-right",
    ],
}

# Clean display labels matching the prompts above (same order)
ATTRIBUTE_LABELS: dict[str, list[str]] = {
    "color": ["red", "blue", "white", "black", "gray", "silver", "green", "yellow", "brown"],
    "body_style": ["sedan", "SUV", "hatchback", "pickup truck", "van", "bus", "motorcycle", "truck"],
    "orientation": [
        "front", "rear", "left side", "right side",
        "front-left", "front-right", "rear-left", "rear-right",
    ],
}

# ── Existing Track 2 artifacts for initial migration ─────────────
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _find_track2_path(env_key: str, subpath: str) -> Path | None:
    if os.environ.get(env_key):
        return Path(os.environ[env_key])
    # Candidate 1: relative to current working directory
    p1 = Path(subpath)
    if p1.exists():
        return p1
    # Candidate 2: relative to ../subpath
    p2 = Path(f"../{subpath}")
    if p2.exists():
        return p2
    # Candidate 3: relative to repo root
    p3 = REPO_ROOT / subpath
    if p3.exists():
        return p3
    return None


TRACK2_DB_PATH: Path | None = _find_track2_path(
    "TRACK2_DB_PATH", "track2/vehicleiq.db"
)
TRACK2_EMBEDDINGS_PATH: Path | None = _find_track2_path(
    "TRACK2_EMBEDDINGS_PATH", "track2/embeddings/vehicle_embeddings.npy"
)
TRACK2_IDS_PATH: Path | None = _find_track2_path(
    "TRACK2_IDS_PATH", "track2/embeddings/vehicle_ids.json"
)
TRACK2_CROPS_BASE_DIR: Path | None = _find_track2_path(
    "TRACK2_CROPS_BASE_DIR", "track2"
)

# ── Track 1 CityFlow-NL benchmark dataset artifacts ──────────────
TRACK1_EMBEDDINGS_PATH: Path | None = _find_track2_path(
    "TRACK1_EMBEDDINGS_PATH", "track1/embeddings/siglip2/vehicle_embeddings.npy"
)
TRACK1_IDS_PATH: Path | None = _find_track2_path(
    "TRACK1_IDS_PATH", "track1/embeddings/siglip2/vehicle_ids.json"
)
TRACK1_METADATA_PATH: Path | None = _find_track2_path(
    "TRACK1_METADATA_PATH", "track1/metadata/train_sampled_8.json"
)

# ── API ───────────────────────────────────────────────────────────
API_URL: str = _str("VEHICLEIQ_API_URL", "http://localhost:8000")


def ensure_dirs() -> None:
    """Create all required storage directories."""
    for d in [DATA_DIR, DB_PATH.parent, FAISS_PATH.parent, VIDEOS_DIR, CROPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
