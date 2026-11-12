#!/usr/bin/env python
"""Score saved WorldTrack predictions of any model with the exact Gate 0 metrics.

Reads ``<pred-dir>/<subset>/<sequence>.npz`` holding ``pred_tracks_ref0 [T, N, 3]`` (tracks of
the frame-0 queries, in camera-0 coordinates, any scale) and ``query_uv [N, 2]``, rebuilds
the GT and the query set with ``eval_track3d_in_worldtrack.py`` itself, checks that the
query pixels match, and computes ``_metrics_for_sequence`` (global median scale, APD over the
fixed thresholds, EPE, dynamic subset) and ``_aggregate_results``. Writes ``summary.json`` and
a ``summary.txt`` in the Gate 0 format. ``--self-test`` scores the GT itself (APD must be 1).
``--in-image`` keeps only queries whose frame-0 pixel lies inside the image, ARROW's query rule
(``-0.5 <= u <= W - 0.5`` and ``-0.5 <= v <= H - 0.5``); upstream has no such bound.

Example (this repo's env, e.g. conda ``d4rt``):
    python scripts/l3d/score_worldtrack_preds.py --data-root data/worldtrack_release \
        --pred-dir output/l3d/baselines/trackeverything --out-dir output/l3d/baselines/trackeverything_scored
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval_track3d_in_worldtrack import (  # noqa: E402
    _aggregate_results,
    _format_subset_summary,
    _metrics_for_sequence,
    load_worldtrack_sequence,
)

SUBSETS = "adt_mini,po_mini,pstudio_mini,ds_mini"


def in_image(uv: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Queries whose pixel is inside an ``hw`` image with half-pixel borders (ARROW's rule)."""
    h, w = hw
    return (uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)


def gt_for(path: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """Upstream query pixels, GT world tracks and the native image ``(H, W)`` (``main`` of the upstream eval)."""
    sample = load_worldtrack_sequence(path, num_frames=num_frames)
    visible = np.asarray(sample["visibility"][0], dtype=bool)
    if not visible.any():
        return None
    uv = np.asarray(sample["tracks_uv"][0, visible], dtype=np.float64)
    depth0 = np.asarray(sample["tracks_xyz_cam"][0, visible, 2], dtype=np.float64)
    keep = np.isfinite(uv).all(-1) & np.isfinite(depth0) & (np.abs(depth0) > 1e-8)
    if not keep.any():
        return None
    gt = np.asarray(sample["tracks_xyz_world"][:, visible], dtype=np.float64)[:, keep]
    return uv[keep], gt, tuple(int(x) for x in sample["video_rgb"].shape[1:3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subsets", default=SUBSETS)
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="score the GT as the prediction (APD must be 1)")
    ap.add_argument("--in-image", action="store_true", help="score only queries whose frame-0 pixel is inside the image")
    args = ap.parse_args()
    if not args.self_test and args.pred_dir is None:
        ap.error("--pred-dir is required unless --self-test")

    summary: dict[str, Any] = {"pred_dir": str(args.pred_dir), "self_test": args.self_test, "in_image": args.in_image,
                               "subsets": {}}
    tags = (" [self-test]" if args.self_test else "") + (" [in-image queries]" if args.in_image else "")
    lines = ["WorldTrack summary (upstream Gate 0 metrics)" + tags]
    for subset in args.subsets.split(","):
        paths = sorted((args.data_root / subset).glob("*.npz"))
        if args.limit_seqs:
            paths = paths[: args.limit_seqs]
        results, missing, off_image = [], [], 0
        for path in paths:
            ref = gt_for(path, args.num_frames)
            if ref is None:
                continue
            uv, gt, hw = ref
            if args.self_test:
                pred = gt.copy()
            else:
                pf = args.pred_dir / subset / f"{path.stem}.npz"
                if not pf.exists():
                    missing.append(path.stem)
                    continue
                with np.load(pf) as z:
                    pred = np.asarray(z["pred_tracks_ref0"], dtype=np.float64)
                    q = np.asarray(z["query_uv"], dtype=np.float64)
                if q.shape != uv.shape or np.abs(q - uv).max() > 1e-2:
                    raise ValueError(f"{subset}/{path.stem}: saved query pixels differ from the upstream queries")
                if pred.shape != gt.shape:
                    raise ValueError(f"{subset}/{path.stem}: pred {pred.shape} vs GT {gt.shape}")
            if args.in_image:
                ok = in_image(uv, hw)
                off_image += int((~ok).sum())
                if not ok.any():
                    continue
                gt, pred = gt[:, ok], pred[:, ok]
            m = _metrics_for_sequence(gt_tracks_world=gt, pred_tracks_ref0=pred, compute_dyn=True)
            m["video_name"] = path.stem
            results.append(m)
        sub = _aggregate_results(results)
        sub["missing_predictions"] = missing
        if args.in_image:
            sub["off_image_queries_dropped"] = off_image
        summary["subsets"][subset] = {**sub, "sequences": results}
        lines.append(_format_subset_summary(subset, sub) + (f" missing={len(missing)}" if missing else "")
                     + (f" off_image_dropped={off_image}" if args.in_image else ""))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
