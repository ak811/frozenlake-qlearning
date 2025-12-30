from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def utc_timestamp() -> str:
    """UTC timestamp suitable for folder names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)

    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    np.save(p, arr)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


def set_global_seed(seed: int) -> np.random.Generator:
    """
    Return a numpy Generator seeded deterministically.
    We keep random state explicit via `rng` instead of relying on global randomness.
    """
    return np.random.default_rng(seed)


def env_bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return x.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
