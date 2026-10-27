"""Read-only access to decoded-query tables (docs/l3d/EXPERIMENTS.md, section 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DESIGNS: dict[str, int] = {"track": 0, "cross": 1, "grid": 2, "static": 3, "selfmatch": 4, "swap": 5}

# Scalar (0-d) keys: name -> kind ("str", "int", "bool").
SCALAR_KEYS: dict[str, str] = {
    "subset": "str",
    "sequence": "str",
    "clip_start": "int",
    "T": "int",
    "has_pose_gt": "bool",
    "model_ckpt": "str",
    "model_config": "str",
}
OPTIONAL_SCALAR_KEYS: dict[str, str] = {"memory_path": "str"}

# Array keys: name -> (shape with symbols T, N, M, dtype kind or exact dtype).
ARRAY_KEYS: dict[str, tuple[tuple[Any, ...], str]] = {
    "hw_native": ((2,), "int"),
    "K_gt": ((3, 3), "float64"),
    "T_w2c_gt": (("T", 4, 4), "float64"),
    "gt_xyz_cam": (("T", "N", 3), "float32"),
    "gt_uv": (("T", "N", 2), "float32"),
    "gt_vis": (("T", "N"), "bool"),
    "is_static": (("N",), "bool"),
    "design": (("M",), "int8"),
    "q_track": (("M",), "int32"),
    "q_parent": (("M",), "int32"),
    "q_u": (("M",), "float32"),
    "q_v": (("M",), "float32"),
    "q_ts": (("M",), "int16"),
    "q_tt": (("M",), "int16"),
    "q_tc": (("M",), "int16"),
    "p_xyz": (("M", 3), "float32"),
    "p_uv": (("M", 2), "float32"),
    "p_vis_logit": (("M",), "float32"),
    "p_conf_logit": (("M",), "float32"),
    "g_xyz": (("M", 3), "float32"),
    "g_uv": (("M", 2), "float32"),
    "g_vis": (("M",), "bool"),
}

# Per-query keys (first axis M), filtered by :func:`select`.
QUERY_KEYS: tuple[str, ...] = tuple(k for k, (shape, _) in ARRAY_KEYS.items() if shape[0] == "M")


def load_table(path: str | Path) -> dict[str, Any]:
    """Load a decoded-query ``.npz``; 0-d entries become Python ``str``/``int``/``bool``."""
    out: dict[str, Any] = {}
    with np.load(Path(path), allow_pickle=False) as npz:
        for key in npz.files:
            arr = npz[key]
            if arr.ndim == 0:
                kind = SCALAR_KEYS.get(key, OPTIONAL_SCALAR_KEYS.get(key))
                if kind == "str" or arr.dtype.kind in ("U", "S"):
                    out[key] = str(arr)
                elif kind == "bool" or arr.dtype.kind == "b":
                    out[key] = bool(arr)
                elif kind == "int" or arr.dtype.kind in ("i", "u"):
                    out[key] = int(arr)
                else:
                    out[key] = arr.item()
            else:
                out[key] = arr
    return out


def _dtype_ok(arr: np.ndarray, want: str) -> bool:
    if want == "int":
        return arr.dtype.kind in ("i", "u")
    return arr.dtype == np.dtype(want)


def validate_table(d: dict[str, Any], strict: bool = True) -> list[str]:
    """Check every required key, shape and dtype of section 3; return the problems found.

    Raises ``ValueError`` listing all problems when ``strict`` and any were found.
    """
    problems: list[str] = []
    for key, kind in SCALAR_KEYS.items():
        if key not in d:
            problems.append(f"missing scalar '{key}'")
            continue
        val = d[key]
        if isinstance(val, np.ndarray):
            if val.ndim != 0:
                problems.append(f"'{key}' should be 0-d, got shape {val.shape}")
                continue
            val = val.item()
        ok = {
            "str": isinstance(val, str),
            "int": isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)),
            "bool": isinstance(val, (bool, np.bool_)),
        }[kind]
        if not ok:
            problems.append(f"'{key}' should be {kind}, got {type(val).__name__}")

    dims: dict[str, int] = {}
    if "T" in d and not isinstance(d["T"], np.ndarray):
        dims["T"] = int(d["T"])
    for key, (shape, want) in ARRAY_KEYS.items():
        if key not in d:
            problems.append(f"missing array '{key}'")
            continue
        arr = np.asarray(d[key])
        if arr.ndim != len(shape):
            problems.append(f"'{key}' should have {len(shape)} dims {shape}, got {arr.shape}")
            continue
        for axis, (sym, got) in enumerate(zip(shape, arr.shape)):
            if isinstance(sym, int):
                if got != sym:
                    problems.append(f"'{key}' axis {axis} should be {sym}, got {got}")
            elif sym in dims and dims[sym] != got:
                problems.append(f"'{key}' axis {axis} ({sym}) is {got}, expected {dims[sym]}")
            else:
                dims.setdefault(sym, got)
        if not _dtype_ok(arr, want):
            problems.append(f"'{key}' dtype should be {want}, got {arr.dtype}")

    # Value-level checks that cheap analysis code relies on.
    if not problems:
        T = dims["T"]
        for key in ("q_ts", "q_tt", "q_tc"):
            a = d[key]
            if a.size and (a.min() < 0 or a.max() >= T):
                problems.append(f"'{key}' out of range [0, {T})")
        q = d["q_track"]
        if q.size and (q.min() < -1 or q.max() >= dims.get("N", 0)):
            problems.append("'q_track' out of range [-1, N)")
        no_track = q == -1
        if np.isfinite(d["g_xyz"][no_track]).any():
            problems.append("'g_xyz' should be NaN where q_track == -1")
        unknown = ~np.isin(d["design"], list(DESIGNS.values()))
        if unknown.any():
            problems.append(f"unknown design ids {np.unique(d['design'][unknown]).tolist()}")
        problems += _parent_problems(d)
    if strict and problems:
        raise ValueError("invalid decoded-query table:\n  " + "\n  ".join(problems))
    return problems


def _parent_problems(d: dict[str, Any]) -> list[str]:
    """Design-5 rows point at a design-4 row ``(n, t1, t2)`` with ``t2`` = their ``q_ts``; others are -1."""
    parent = d["q_parent"].astype(np.int64)
    is_swap = d["design"] == DESIGNS["swap"]
    if np.any(parent[~is_swap] != -1):
        return ["'q_parent' should be -1 outside design 5"]
    p = parent[is_swap]
    if p.size == 0:
        return []
    if p.min() < 0 or p.max() >= parent.shape[0]:
        return ["'q_parent' out of range [0, M) in design 5"]
    ok = (
        (d["design"][p] == DESIGNS["selfmatch"])
        & (d["q_track"][p] == d["q_track"][is_swap])
        & (d["q_tt"][p] == d["q_ts"][is_swap])
        & (d["q_tc"][p] == d["q_ts"][is_swap])
    )
    if not ok.all():
        return [f"{int(np.count_nonzero(~ok))} design-5 rows do not match their design-4 parent"]
    return []


def select(
    d: dict[str, Any],
    design: int | str | None = None,
    self_frame: bool | None = None,
    has_gt: bool | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Row filter: returns a shallow copy with per-query (``M``) arrays filtered.

    ``design`` is an id or a name from :data:`DESIGNS`; ``self_frame=True`` keeps
    ``q_tc == q_tt`` (``False`` the opposite); ``has_gt=True`` keeps finite ``g_xyz``.
    The original row indices are stored under ``row_index``; ``q_parent`` keeps
    pointing at rows of the original table.
    """
    m = int(np.asarray(d["design"]).shape[0])
    keep = np.ones(m, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if design is not None:
        did = DESIGNS[design] if isinstance(design, str) else int(design)
        keep &= d["design"] == did
    if self_frame is not None:
        is_self = d["q_tc"] == d["q_tt"]
        keep &= is_self if self_frame else ~is_self
    if has_gt is not None:
        fin = np.isfinite(d["g_xyz"]).all(-1)
        keep &= fin if has_gt else ~fin
    out = dict(d)
    for key in QUERY_KEYS:
        if key in d:
            out[key] = d[key][keep]
    prev = d.get("row_index", np.arange(m))
    out["row_index"] = np.asarray(prev)[keep]
    return out


def source_swap_rows(d: dict[str, Any]) -> dict[str, np.ndarray]:
    """Row indices of every source-swap triple in a full table (not a :func:`select` output).

    For each design-5 row ``b = P(u2_hat, t2, t3, c)``: ``a`` = design-1 row
    ``P(u1, t1, t3, c)`` (the reference, ``t1`` = the parent's ``q_ts``), ``g`` = design-1
    row ``P(u2_gt, t2, t3, c)`` (GT-correspondence control), ``parent`` = design-4 row
    whose ``p_uv`` is ``u2_hat``. Triples without both design-1 partners are dropped.
    Also returns ``n, t1, t2, t3, c``.
    """
    if "row_index" in d:
        raise ValueError("source_swap_rows needs the full table; q_parent indexes its rows.")
    T = int(d["T"])
    n_all, ts_all, tt_all, tc_all = (d[k].astype(np.int64) for k in ("q_track", "q_ts", "q_tt", "q_tc"))

    def key(n: np.ndarray, s: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
        return ((n * T + s) * T + t) * T + c

    d1 = np.flatnonzero(d["design"] == DESIGNS["cross"])
    k1 = key(n_all[d1], ts_all[d1], tt_all[d1], tc_all[d1])
    order = np.argsort(k1, kind="stable")
    k1, d1 = k1[order], d1[order]

    def find(k: np.ndarray) -> np.ndarray:
        if k1.size == 0:
            return np.full(k.shape, -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(k1, k), 0, k1.size - 1)
        return np.where(k1[pos] == k, d1[pos], -1)

    b = np.flatnonzero(d["design"] == DESIGNS["swap"])
    parent = d["q_parent"][b].astype(np.int64)
    n, t1, t2, t3, c = n_all[b], ts_all[parent], ts_all[b], tt_all[b], tc_all[b]
    a = find(key(n, t1, t3, c))
    g = find(key(n, t2, t3, c))
    keep = (a >= 0) & (g >= 0)
    out = {"a": a, "b": b, "g": g, "parent": parent, "n": n, "t1": t1, "t2": t2, "t3": t3, "c": c}
    return {k: v[keep] for k, v in out.items()}
