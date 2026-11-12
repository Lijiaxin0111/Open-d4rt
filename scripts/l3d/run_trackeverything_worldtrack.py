#!/usr/bin/env python
"""Run TrackEverything on WorldTrack with the upstream Gate 0 queries; saves one .npz per sequence.

Runs in TrackEverything's own environment (github.com/ayushjain1144/trackeverything, weights
``katefgroup/TrackEverything``), with only numpy and cv2 from this repo's side. Queries are
exactly those of ``eval_track3d_in_worldtrack.py``: the first ``--num-frames`` frames, tracks
visible at frame 0 with a finite projected pixel and depth, queried at frame 0 in native
pixels. TrackEverything runs in sparse mode with its default geometry (VGGT-Omega, RGB only),
whose world frame is camera 0. Each output holds ``pred_tracks_ref0 [T, N, 3]``,
``pred_visibs [T, N]`` and the query pixels; score them with ``score_worldtrack_preds.py``.

Example (in TrackEverything's env; its repo directory is the ``trackeverything`` package, so its
parent goes on PYTHONPATH):
    PYTHONPATH=/path/to/parent_of_trackeverything python /path/to/l3d-d4rt/scripts/l3d/run_trackeverything_worldtrack.py \
        --data-root /path/to/worldtrack_release --out-dir /path/to/preds/trackeverything
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

SUBSETS = "adt_mini,po_mini,pstudio_mini,ds_mini"


def load_sequence(npz_path: Path, num_frames: int) -> dict[str, np.ndarray]:
    """Frames and frame-0 queries as in ``eval_track3d_in_worldtrack.load_worldtrack_sequence`` + ``main``."""
    pack = np.load(npz_path, allow_pickle=True)
    jpeg = np.asarray(pack["images_jpeg_bytes"])
    xyz = np.asarray(pack["tracks_XYZ"], dtype=np.float64)
    vis = np.asarray(pack["visibility"], dtype=bool)
    fx, fy, cx, cy = np.asarray(pack["fx_fy_cx_cy"], dtype=np.float64).reshape(-1)[:4]
    n = min(int(num_frames), jpeg.shape[0], xyz.shape[0], vis.shape[0])
    frames = np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                       for b in jpeg[:n]])
    z = xyz[0, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_z = np.where(np.abs(z) > 1e-8, z, np.nan)
        uv = np.stack([xyz[0, :, 0] / safe_z * fx + cx, xyz[0, :, 1] / safe_z * fy + cy], -1)
    visible = vis[0]
    keep = np.isfinite(uv[visible]).all(-1) & np.isfinite(z[visible]) & (np.abs(z[visible]) > 1e-8)
    return {"frames": frames, "query_uv": uv[visible][keep], "n_frames": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subsets", default=SUBSETS)
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--model", default="katefgroup/TrackEverything")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--geometry-backend", default="pi3", choices=["pi3", "vggt_omega"],
                    help="vggt_omega needs a local checkpoint that is not public (cluster path); pi3 downloads from HF")
    args = ap.parse_args()

    from trackeverything.demo_hf import TrackEverything  # noqa: PLC0415 - TrackEverything's env only

    model = TrackEverything.from_pretrained(args.model, device=args.device, geometry_backend=args.geometry_backend)
    log = []
    for subset in args.subsets.split(","):
        paths = sorted((args.data_root / subset).glob("*.npz"))
        if args.limit_seqs:
            paths = paths[: args.limit_seqs]
        (args.out_dir / subset).mkdir(parents=True, exist_ok=True)
        for path in paths:
            out = args.out_dir / subset / f"{path.stem}.npz"
            if out.exists():
                continue
            seq = load_sequence(path, args.num_frames)
            if seq["query_uv"].shape[0] == 0:
                print(f"[te] skip {subset}/{path.stem}: no frame-0 queries", flush=True)
                continue
            q = np.concatenate([np.zeros((seq["query_uv"].shape[0], 1)), seq["query_uv"]], axis=1)  # [t=0, x, y]
            start = time.time()
            res = model.infer(video=seq["frames"], query_points_2d=q, mode="sparse")
            coords = np.asarray(res.query_coords.detach().float().cpu())[0]  # [T, N, 3], world = camera 0
            visibs = np.asarray(res.query_visibs.detach().float().cpu())[0]
            secs = time.time() - start
            ext = getattr(res, "extrinsics", None)
            cam0_dev = (float(np.abs(np.asarray(ext.detach().float().cpu()).reshape(-1, 4, 4)[0] - np.eye(4)).max())
                        if ext is not None else float("nan"))  # world must be camera 0
            np.savez_compressed(out, pred_tracks_ref0=coords.astype(np.float32), pred_visibs=visibs.astype(np.float32),
                                query_uv=seq["query_uv"].astype(np.float64), norm_factor=float(getattr(res, "norm_factor", 1.0)),
                                seconds=secs)
            log.append({"subset": subset, "sequence": path.stem, "queries": int(q.shape[0]), "frames": seq["n_frames"],
                        "pred_frames": int(coords.shape[0]), "seconds": secs,
                        "geometry_backend": args.geometry_backend, "cam0_extrinsic_dev": cam0_dev})
            print(f"[te] {subset}/{path.stem}: {q.shape[0]} queries, {coords.shape} preds, {secs:.1f}s, cam0 dev {cam0_dev:.2e}", flush=True)
    (args.out_dir / "run_log.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
