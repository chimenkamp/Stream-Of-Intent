from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from stream_of_intent.model_representation import AbstractModelRepresentation
from stream_of_intent.types import FEATURE_ORDER, FeatureVector

_DEFAULT_DB = Path(__file__).resolve().parent / "models.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    description     TEXT    DEFAULT '',
    model_json      TEXT    NOT NULL,
    spec_json       TEXT,
    best_distance   REAL,
    achieved_features TEXT,
    static_params   TEXT,
    created_at      REAL    NOT NULL,
    tags            TEXT    DEFAULT ''
);
"""


@dataclass
class ModelRecord:
    id: int
    name: str
    description: str
    model_json: str
    spec_json: Optional[str]
    best_distance: Optional[float]
    achieved_features: Optional[str]
    static_params: Optional[str]
    created_at: float
    tags: str


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = str(db_path or _DEFAULT_DB)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def save_model(
    name: str,
    representation: AbstractModelRepresentation,
    *,
    description: str = "",
    tags: str = "",
    db_path: Path | str | None = None,
) -> int:
    model_json = representation.to_json()
    meta = representation.metadata or {}
    spec_json = json.dumps(meta.get("spec")) if meta.get("spec") else None
    best_distance = meta.get("best_distance")
    static_json = json.dumps(representation.static_params.__dict__) if representation.static_params else None

    achieved = None
    if "achieved_features" in meta:
        achieved = json.dumps(meta["achieved_features"])

    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO models
               (name, description, model_json, spec_json, best_distance,
                achieved_features, static_params, created_at, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, description, model_json, spec_json, best_distance,
             achieved, static_json, time.time(), tags),
        )
        return cur.lastrowid


def list_models(db_path: Path | str | None = None) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, description, best_distance, created_at, tags FROM models ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_model(model_id: int, db_path: Path | str | None = None) -> Optional[ModelRecord]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if row is None:
        return None
    return ModelRecord(**dict(row))


def get_model_by_name(name: str, db_path: Path | str | None = None) -> Optional[ModelRecord]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
    if row is None:
        return None
    return ModelRecord(**dict(row))


def load_representation(model_id: int, db_path: Path | str | None = None) -> Optional[AbstractModelRepresentation]:
    record = get_model(model_id, db_path)
    if record is None:
        return None
    return AbstractModelRepresentation.from_json(record.model_json)


def delete_model(model_id: int, db_path: Path | str | None = None) -> bool:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        return cur.rowcount > 0


def update_model(
    model_id: int,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,
    db_path: Path | str | None = None,
) -> bool:
    sets = []
    values: list = []
    if name is not None:
        sets.append("name = ?")
        values.append(name)
    if description is not None:
        sets.append("description = ?")
        values.append(description)
    if tags is not None:
        sets.append("tags = ?")
        values.append(tags)
    if not sets:
        return False
    values.append(model_id)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE models SET {', '.join(sets)} WHERE id = ?", values,
        )
        return cur.rowcount > 0


def import_model_json(
    name: str,
    json_str: str,
    *,
    description: str = "",
    tags: str = "",
    db_path: Path | str | None = None,
) -> int:
    rep = AbstractModelRepresentation.from_json(json_str)
    return save_model(name, rep, description=description, tags=tags, db_path=db_path)
