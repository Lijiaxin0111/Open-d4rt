#!/usr/bin/env python
"""Run the unmodified upstream WorldTrack eval (Gate 0) and also save OpenD4RT's predictions.

Wraps ``eval_track3d_in_worldtrack.main``. Every argument except ``--pred-dir`` goes to upstream
unchanged. Two module-level hooks record each sequence's query pixels and the exact
``pred_tracks_ref0 [T, N, 3]`` (camera-0 coordinates) that upstream scores, as
``<pred-dir>/<subset>/<sequence>.npz`` in the format of ``score_worldtrack_preds.py``, so
OpenD4RT can be rescored with other query rules (e.g. ``--in-image``) without rerunning the
model. Upstream still writes its own summary, so the run also re-checks Gate 0.

Example (this repo's env on the GPU box; arguments as in ``scripts/l3d/run_gate0.sh``):
    python scripts/l3d/run_opend4rt_worldtrack_preds.py --pred-dir output/l3d/baselines/opend4rt \
        --model-config checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
        --ckpt-path checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
        --data-root data/worldtrack_release --num-frames 64 --query-chunk-size 4096 \
        --output-dir output/gate0_48clip_rerun --device cuda --save-per-sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import eval_track3d_in_worldtrack as upstream  # noqa: E402


def upstream_query_uv(sample: dict[str, Any]) -> np.ndarray:
    """The query pixels ``main`` builds: frame-0 visible tracks with a finite pixel and depth."""
    vis = np.asarray(sample["visibility"][0], dtype=bool)
    uv = np.asarray(sample["tracks_uv"][0, vis], dtype=np.float64)
    depth0 = np.asarray(sample["tracks_xyz_cam"][0, vis, 2], dtype=np.float64)
    keep = np.isfinite(uv).all(-1) & np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    return uv[keep]


def install_hooks(pred_dir: Path) -> None:
    """Patch ``upstream``'s loader and per-sequence metric so each scored prediction is saved."""
    load, metrics = upstream.load_worldtrack_sequence, upstream._metrics_for_sequence
    current: dict[str, Any] = {}

    def load_hook(npz_path: Path, num_frames: int) -> dict[str, Any]:
        sample = load(npz_path, num_frames=num_frames)
        current.update(path=Path(npz_path), sample=sample)
        return sample

    def metrics_hook(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pred = np.asarray(kwargs["pred_tracks_ref0"])
        uv = upstream_query_uv(current["sample"])
        if uv.shape[0] != pred.shape[1]:
            raise RuntimeError(f"{current['path'].name}: {uv.shape[0]} query pixels vs {pred.shape[1]} predictions")
        out = pred_dir / current["path"].parent.name / f"{current['path'].stem}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, pred_tracks_ref0=pred.astype(np.float32), query_uv=uv)
        return metrics(*args, **kwargs)

    upstream.load_worldtrack_sequence = load_hook
    upstream._metrics_for_sequence = metrics_hook


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ours, rest = ap.parse_known_args()
    install_hooks(ours.pred_dir)
    sys.argv = [sys.argv[0], *rest]
    return upstream.main()


if __name__ == "__main__":
    raise SystemExit(main())
