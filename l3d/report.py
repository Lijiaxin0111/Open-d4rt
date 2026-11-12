"""Shared helpers for the Part A runners in ``scripts/l3d`` (dump iteration, JSON output)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from l3d.table import load_table

SUBSETS = ("adt_mini", "po_mini", "pstudio_mini", "ds_mini")
KNOWN_MODELS = {"OpenD4RT_48CLIP": "OpenD4RT 48CLIP", "flow3r": "flow3r"}  # dump run_name prefix -> report label


def model_label(dump: Path, override: str | None = None) -> str:
    """Report label of the model behind a dump: ``override``, else from its manifest's ``run_name``."""
    if override:
        return override
    try:
        name = json.loads((Path(dump) / "manifest.json").read_text()).get("run_name") or Path(dump).name
    except (OSError, ValueError):
        name = Path(dump).name
    return next((label for key, label in KNOWN_MODELS.items() if name.startswith(key)), name)


def jsonable(x: Any) -> Any:
    """Numpy-free copy of ``x`` for ``json.dumps`` (non-finite floats become ``None``)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def iter_clips(dump: Path, subsets: list[str], limit: int = 0) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """``(manifest entry, table)`` for every finished clip of ``subsets`` (``limit`` per subset)."""
    manifest = json.loads((Path(dump) / "manifest.json").read_text())
    seen = {s: 0 for s in subsets}
    for entry in manifest["clips"]:
        s = entry["subset"]
        if s not in seen or entry.get("status") != "ok" or (limit and seen[s] >= limit):
            continue
        seen[s] += 1
        yield entry, load_table(Path(dump) / entry["file"])


def fmt(x: float | None, p: int = 3) -> str:
    """Markdown cell for a float (``n/a`` when missing or non-finite)."""
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"
