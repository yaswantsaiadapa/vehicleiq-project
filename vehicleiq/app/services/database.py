"""
VehicleIQ — DatabaseService
=============================
Thread-safe SQLite wrapper.  Handles schema creation, the one-time
migration of existing Track 2 prototype data, and all CRUD operations
used by the processing pipeline and API routers.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────────────────────────

_DDL_VEHICLES = """
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id              TEXT PRIMARY KEY,
    video_id                TEXT,
    track_id                INTEGER,
    first_frame             INTEGER,
    last_frame              INTEGER,
    frame_count             INTEGER,
    vehicle_type            TEXT,
    vehicle_type_confidence REAL,
    color                   TEXT,
    color_confidence        REAL,
    body_style              TEXT,
    body_style_confidence   REAL,
    orientation             TEXT,
    orientation_confidence  REAL,
    mean_speed              REAL,
    median_speed            REAL,
    max_speed               REAL,
    embedding_index         INTEGER,
    json_record             TEXT,
    crop_base_dir           TEXT
);
"""

_DDL_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    video_id            TEXT PRIMARY KEY,
    original_filename   TEXT,
    status              TEXT DEFAULT 'queued',
    progress_message    TEXT,
    fps                 REAL,
    frame_count         INTEGER,
    width               INTEGER,
    height              INTEGER,
    vehicle_count       INTEGER DEFAULT 0,
    crop_base_dir       TEXT,
    created_at          TEXT,
    completed_at        TEXT,
    error_message       TEXT
);
"""

_DDL_ADD_CROP_BASE = """
ALTER TABLE vehicles ADD COLUMN crop_base_dir TEXT;
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseService:
    """
    Manages a single SQLite database for VehicleIQ.

    Uses WAL mode and a per-thread connection cache so that background
    processing threads can safely share the database.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or config.DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()
        self._maybe_migrate_track2()
        self._maybe_migrate_track1()

    # ── Connection ────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    # ── Schema ────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute(_DDL_VEHICLES)
        conn.execute(_DDL_VIDEOS)
        # Add crop_base_dir column to existing databases that lack it
        try:
            conn.execute(_DDL_ADD_CROP_BASE)
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
        log.info("Schema initialised at %s", self._db_path)

    # ── Track 2 migration ─────────────────────────────────────────

    def _maybe_migrate_track2(self) -> None:
        """
        On the very first startup, if the source Track 2 database
        exists and the application DB has no rows, copy the vehicles
        across and register the source video.
        """
        if config.TRACK2_DB_PATH is None or not config.TRACK2_DB_PATH.exists():
            return

        conn = self._conn()
        cur = conn.execute("SELECT COUNT(*) FROM vehicles")
        if cur.fetchone()[0] > 0:
            return  # already migrated

        log.info("Migrating Track 2 prototype data from %s", config.TRACK2_DB_PATH)

        crops_base = str(config.TRACK2_CROPS_BASE_DIR.resolve()) if config.TRACK2_CROPS_BASE_DIR else None

        # Register the source video
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
                (video_id, original_filename, status, vehicle_count,
                 crop_base_dir, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "S01_c001",
                "vdo.avi",
                "complete",
                165,
                crops_base,
                _utcnow(),
                _utcnow(),
            ),
        )

        # Copy vehicle rows, updating crop_base_dir
        src_conn = sqlite3.connect(str(config.TRACK2_DB_PATH))
        src_conn.row_factory = sqlite3.Row
        try:
            for row in src_conn.execute("SELECT * FROM vehicles"):
                d = dict(row)
                # Add/update crop_base_dir
                d["crop_base_dir"] = crops_base
                # Re-parse json_record to fix attribute display labels
                if d.get("json_record"):
                    try:
                        record = json.loads(d["json_record"])
                        record = _clean_vehicle_record(record)
                        d["json_record"] = json.dumps(record)
                    except Exception:
                        pass
                conn.execute(
                    """
                    INSERT OR IGNORE INTO vehicles
                        (vehicle_id, video_id, track_id, first_frame, last_frame,
                         frame_count, vehicle_type, vehicle_type_confidence, color,
                         color_confidence, body_style, body_style_confidence,
                         orientation, orientation_confidence, mean_speed, median_speed,
                         max_speed, embedding_index, json_record, crop_base_dir)
                    VALUES
                        (:vehicle_id, :video_id, :track_id, :first_frame, :last_frame,
                         :frame_count, :vehicle_type, :vehicle_type_confidence, :color,
                         :color_confidence, :body_style, :body_style_confidence,
                         :orientation, :orientation_confidence, :mean_speed, :median_speed,
                         :max_speed, :embedding_index, :json_record, :crop_base_dir)
                    """,
                    d,
                )
        finally:
            src_conn.close()

        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        log.info("Migration complete — %d vehicles imported", count)

    def _maybe_migrate_track1(self) -> None:
        """
        Migrate the full CityFlow-NL 2,155 vehicles from Track 1 metadata
        into the SQLite vehicles table so search results are fully enriched
        with color, type, descriptions, and camera metadata.
        """
        if config.TRACK1_METADATA_PATH is None or not config.TRACK1_METADATA_PATH.exists():
            return

        conn = self._conn()
        cur = conn.execute("SELECT COUNT(*) FROM vehicles")
        if cur.fetchone()[0] >= 2000:
            return  # already migrated full dataset

        log.info("Migrating Track 1 benchmark dataset (2,155 vehicles) into database...")

        # Register benchmark video summary entry
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
                (video_id, original_filename, status, vehicle_count,
                 fps, frame_count, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cityflow_benchmark",
                "CityFlow-NL Benchmark (23 Surveillance Cameras)",
                "complete",
                2155,
                10.0,
                65077,
                _utcnow(),
                _utcnow(),
            ),
        )

        with open(config.TRACK1_METADATA_PATH, "r", encoding="utf-8") as f:
            t1_meta = json.load(f)

        known_colors = [
            "red", "blue", "white", "black", "gray", "silver", "green",
            "yellow", "brown", "maroon", "grey", "dark"
        ]
        known_types = [
            "sedan", "suv", "pickup truck", "pickup", "truck", "van", "bus",
            "motorcycle", "hatchback", "car", "wagon", "flatbed", "semi-truck"
        ]

        batch = []
        for vid, rec in t1_meta.items():
            descriptions = rec.get("descriptions", [])
            other_views = rec.get("descriptions_other_views", [])
            all_text = " ".join(descriptions + other_views).lower()

            # Attribute parsing from human descriptions
            color = next((c for c in known_colors if c in all_text), "unknown")
            if color == "grey":
                color = "gray"
            vtype = next((t for t in known_types if t in all_text), "car")
            if vtype in ("pickup truck", "pickup"):
                vtype = "pickup truck"
            elif vtype in ("semi-truck", "flatbed"):
                vtype = "truck"

            frames = rec.get("source", {}).get("frames", [])
            frame_count = len(frames)
            camera = "c001"
            if frames:
                parts = frames[0].replace("\\", "/").split("/")
                for p in parts:
                    if p.startswith("c0") or p.startswith("c1"):
                        camera = p
                        break

            rep_crops = [
                img["crop_path"].replace("\\", "/").split("/")[-1]
                for img in rec.get("images", [])
                if "crop_path" in img
            ]

            json_rec = {
                "vehicle_id": str(vid),
                "source": {"video_id": f"cityflow_{camera}"},
                "tracking": {"frame_count": frame_count, "first_frame": 1, "last_frame": frame_count},
                "vehicle": {
                    "type": {"value": vtype, "score": 1.0},
                    "color": {"value": color, "score": 1.0},
                    "body_style": {"value": vtype, "score": 1.0},
                },
                "descriptions": descriptions,
                "visual": {"representative_crops": rep_crops},
            }

            batch.append((
                str(vid),
                f"cityflow_{camera}",
                0,
                1,
                frame_count,
                frame_count,
                vtype,
                1.0,
                color,
                1.0,
                vtype,
                1.0,
                "unspecified",
                1.0,
                None,
                None,
                None,
                None,
                json.dumps(json_rec),
                "",
            ))

        conn.executemany(
            """
            INSERT OR IGNORE INTO vehicles
                (vehicle_id, video_id, track_id, first_frame, last_frame,
                 frame_count, vehicle_type, vehicle_type_confidence, color,
                 color_confidence, body_style, body_style_confidence,
                 orientation, orientation_confidence, mean_speed, median_speed,
                 max_speed, embedding_index, json_record, crop_base_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        log.info("Track 1 migration complete — %d total vehicles now in database", count)

    # ── Videos ───────────────────────────────────────────────────

    def insert_video(
        self,
        video_id: str,
        original_filename: str,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
        crop_base_dir: str,
    ) -> None:
        with self._write_lock:
            self._conn().execute(
                """
                INSERT INTO videos
                    (video_id, original_filename, status, fps, frame_count,
                     width, height, crop_base_dir, created_at)
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (video_id, original_filename, fps, frame_count, width, height,
                 crop_base_dir, _utcnow()),
            )
            self._conn().commit()

    def update_video_status(
        self,
        video_id: str,
        status: str,
        progress_message: str = "",
        vehicle_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        params: list[Any] = [status, progress_message]
        if vehicle_count is not None:
            params.append(vehicle_count)
        if status in ("complete", "failed"):
            params.append(_utcnow())
        if error_message is not None:
            params.append(error_message)
        params.append(video_id)

        # Build the SET clause dynamically
        sets = ["status = ?", "progress_message = ?"]
        if vehicle_count is not None:
            sets.append("vehicle_count = ?")
        if status in ("complete", "failed"):
            sets.append("completed_at = ?")
        if error_message is not None:
            sets.append("error_message = ?")

        sql = f"UPDATE videos SET {', '.join(sets)} WHERE video_id = ?"
        with self._write_lock:
            self._conn().execute(sql, params)
            self._conn().commit()

    def get_video(self, video_id: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_videos(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM videos ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Vehicles ──────────────────────────────────────────────────

    def insert_vehicle(
        self,
        vehicle_id: str,
        video_id: str,
        track_id: int,
        first_frame: int,
        last_frame: int,
        frame_count: int,
        vehicle_type: str,
        vehicle_type_score: float,
        color: str,
        color_score: float,
        body_style: str,
        body_style_score: float,
        orientation: str,
        orientation_score: float,
        mean_speed: float | None,
        median_speed: float | None,
        max_speed: float | None,
        embedding_index: int,
        json_record: dict,
        crop_base_dir: str,
    ) -> None:
        with self._write_lock:
            self._conn().execute(
                """
                INSERT OR REPLACE INTO vehicles
                    (vehicle_id, video_id, track_id, first_frame, last_frame,
                     frame_count, vehicle_type, vehicle_type_confidence, color,
                     color_confidence, body_style, body_style_confidence,
                     orientation, orientation_confidence, mean_speed, median_speed,
                     max_speed, embedding_index, json_record, crop_base_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vehicle_id, video_id, track_id, first_frame, last_frame,
                    frame_count, vehicle_type, vehicle_type_score, color,
                    color_score, body_style, body_style_score, orientation,
                    orientation_score, mean_speed, median_speed, max_speed,
                    embedding_index, json.dumps(json_record), crop_base_dir,
                ),
            )
            self._conn().commit()

    def get_vehicle(self, vehicle_id: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("json_record"):
            try:
                d["json_record"] = json.loads(d["json_record"])
            except Exception:
                pass
        return d

    def get_vehicles_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn().execute(
            f"SELECT * FROM vehicles WHERE vehicle_id IN ({placeholders})", ids
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("json_record"):
                try:
                    d["json_record"] = json.loads(d["json_record"])
                except Exception:
                    pass
            result.append(d)
        return result

    def get_vehicles_for_video(self, video_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM vehicles WHERE video_id = ? ORDER BY track_id",
            (video_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("json_record"):
                try:
                    d["json_record"] = json.loads(d["json_record"])
                except Exception:
                    pass
            result.append(d)
        return result

    def vehicle_count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]


# ── Helpers ───────────────────────────────────────────────────────

def _clean_attribute_value(raw: str) -> str:
    """
    Strip prompt prefixes from raw SigLIP attribute values stored in
    the Track 2 prototype records.

    Examples:
        "a red vehicle"                        → "red"
        "a pickup truck"                       → "pickup truck"
        "a vehicle viewed from the front-left" → "front-left"
        "an SUV"                               → "SUV"
    """
    prefixes = [
        "a vehicle viewed from the ",
        "a vehicle viewed from ",
        "a vehicle ",
        "an ",
        "a ",
    ]
    s = raw.strip()
    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):]
            break
    return s


def _clean_vehicle_record(record: dict) -> dict:
    """Clean raw prompt strings inside a legacy vehicle_record dict."""
    vehicle = record.get("vehicle", {})
    for attr in ("color", "body_style", "orientation"):
        if attr in vehicle and "value" in vehicle[attr]:
            vehicle[attr]["value"] = _clean_attribute_value(
                vehicle[attr]["value"]
            )
            # rename 'confidence' → 'score' for new terminology
            if "confidence" in vehicle[attr]:
                vehicle[attr]["score"] = vehicle[attr].pop("confidence")
    return record
