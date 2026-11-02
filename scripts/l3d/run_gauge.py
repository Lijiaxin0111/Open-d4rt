#!/usr/bin/env python
"""GAUGE baseline (a reference with GT anchors, not label-free) on the dump; writes gauge.{json,md}.

Per clip, on the design-0 world-frame outputs (``t_c = 0``, every ``t_t``, scaled by the
clip scale): :func:`l3d.gauge.gauge_correct` and the per-group Sim(3) reference with
1% and 5% anchor tracks, evaluated with the upstream APD thresholds and EPE on the
non-anchor points (all, and dynamic = total GT motion > 0.01 as upstream), averaged
over clips like the upstream per-sequence summary. No re-alignment after correction.

Example:
    .venv/bin/python scripts/l3d/run_gauge.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import check_rows as cr  # noqa: E402
from l3d import gauge  # noqa: E402
from l3d.decompose import group_tracks, table_scale, world_trajectories  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable  # noqa: E402
from l3d.table import select  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
FRACS = (0.01, 0.05)
VARIANTS = ("base", "gauge", "sim3")
# GAUGE Table A1, D4RT (OpenD4RT), EPE improvement % (their grouping and full sequences, not comparable 1:1).
PAPER_EPE_GAIN = {"adt_mini": (24.7, 29.5), "po_mini": (43.4, 18.2), "pstudio_mini": (51.8, 25.6), "ds_mini": (31.9, -8.3)}


def world_arrays(d: dict[str, Any], P_all: np.ndarray) -> dict[str, np.ndarray]:
    """Design-0 ``t_c = 0`` rows as ``[T, N0, ...]`` arrays over the frame-0-visible tracks."""
    r = select(d, design="track", mask=d["q_tc"] == 0)
    tracks, col = np.unique(r["q_track"], return_inverse=True)
    T, n0 = int(d["T"]), tracks.size
    P = np.full((T, n0, 3), np.nan)
    Q = np.full((T, n0, 3), np.nan)
    vis = np.zeros((T, n0), dtype=bool)
    tt = r["q_tt"].astype(np.int64)
    P[tt, col] = P_all[r["row_index"]]
    Q[tt, col] = r["g_xyz"]
    vis[tt, col] = r["p_vis_logit"] > 0
    gt_vis = np.asarray(d["gt_vis"])[:, tracks]
    return {"tracks": tracks, "P": P, "Q": Q, "known": vis & gt_vis & np.isfinite(Q).all(-1)}


def clip_metrics(d: dict[str, Any], seed: int) -> dict[str, Any]:
    err = cr.row_errors(d, table_scale(d))
    w = world_arrays(d, err["P"])
    if w["tracks"].size == 0:
        return {}
    T_w2c = d["T_w2c_gt"]
    z = np.asarray(d["gt_xyz_cam"], dtype=np.float64)[..., 2]
    labels = group_tracks(world_trajectories(d["gt_xyz_cam"], T_w2c), d["is_static"], mode="rigid",
                          ref_depth=float(np.median(z[np.isfinite(z)])))[w["tracks"]]
    pb = cr.pose_baselines(d, err)
    centers = pb["poses_to_0"]["umeyama"][:, :3, 3]
    P, Q, known = w["P"], w["Q"], w["known"]
    with np.errstate(invalid="ignore"):
        motion = np.nanmean(np.linalg.norm(P - P[:1], axis=-1), axis=0)
        dyn = np.nansum(np.linalg.norm(np.diff(Q, axis=0), axis=-1), axis=0) > 0.01
    finite = np.isfinite(Q).all(-1)
    out: dict[str, Any] = {"n_tracks": int(w["tracks"].size), "n_groups": int(np.unique(labels[labels > 0]).size)}
    rng = np.random.default_rng(seed)
    for frac in FRACS:
        anchors = gauge.allocate_anchors(labels, motion, known.any(0), frac, rng)
        preds = {"base": P, "gauge": gauge.gauge_correct(P, Q, known, labels, anchors, centers),
                 "sim3": gauge.sim3_per_group(P, Q, known, labels, anchors)}
        ev = finite & ~anchors[None]
        out[str(frac)] = {
            "n_anchors": int(anchors.sum()),
            **{v: {"all": gauge.apd_epe(p, Q, ev), "dyn": gauge.apd_epe(p, Q, ev & dyn[None])} for v, p in preds.items()},
        }
    return out


def pool(clips: list[dict[str, Any]]) -> dict[str, Any]:
    clips = [c for c in clips if c]
    out: dict[str, Any] = {"n_clips": len(clips)}
    for frac in map(str, FRACS):
        out[frac] = {"n_anchors_mean": float(np.mean([c[frac]["n_anchors"] for c in clips]))}
        for v in VARIANTS:
            for part in ("all", "dyn"):
                vals = [c[frac][v][part] for c in clips if c[frac][v][part]["n"]]
                out[frac].setdefault(v, {})[part] = {k: float(np.mean([x[k] for x in vals])) for k in ("apd", "epe")}
    return out


def write_md(per_subset: dict[str, dict[str, Any]], path: Path, argv: list[str], runtime_s: float) -> None:
    lines = [
        "# GAUGE baseline (GT anchors; a reference, not label-free), OpenD4RT 48CLIP, clip [0, 48)",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s). World-frame outputs (design 0, `t_c = 0`) scaled by the clip's global",
        "median scale; GAUGE (`l3d/gauge.py`: per-group per-frame ray scale about the model's camera centers + median",
        "translation) and a per-group Sim(3) from the same anchors. Groups = GT motion groups (the paper's oracle",
        "variant); anchors = whole GT tracks. Metrics on non-anchor points, upstream APD thresholds, no re-alignment,",
        "mean over clips. Dynamic = total GT motion > 0.01 (upstream rule).",
        "",
        "| Subset | Anchors | Points | APD base | APD GAUGE | APD Sim(3)/group | EPE base | EPE GAUGE | EPE Sim(3)/group "
        "| EPE gain GAUGE | Paper EPE gain (DQS / Full) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s, r in per_subset.items():
        if not r.get("n_clips"):
            continue
        for frac in map(str, FRACS):
            f = r[frac]
            for part in ("all", "dyn"):
                b, g, m = f["base"][part], f["gauge"][part], f["sim3"][part]
                gain = 100 * (b["epe"] - g["epe"]) / b["epe"] if b["epe"] > 0 else float("nan")
                paper = PAPER_EPE_GAIN.get(s, (float("nan"),) * 2)
                lines.append(
                    f"| `{s}` | {float(frac):.0%} ({f['n_anchors_mean']:.0f}/clip) | {part} | {fmt(b['apd'])} | "
                    f"{fmt(g['apd'])} | {fmt(m['apd'])} | {fmt(b['epe'], 4)} | {fmt(g['epe'], 4)} | {fmt(m['epe'], 4)} | "
                    f"{fmt(gain, 1)}% | {paper[0]:+.1f}% / {paper[1]:+.1f}% |"
                )
    lines += ["", "Paper column: GAUGE Table A1 for D4RT (their unsupervised groups, 5% anchors, full sequences); for",
              "orientation only."]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "results/l3d/gauge")
    args = ap.parse_args()

    subsets = args.subsets.split(",")
    start = time.time()
    clips: dict[str, list[dict[str, Any]]] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        clips[entry["subset"]].append(clip_metrics(d, args.seed + zlib.crc32(entry["file"].encode())))
        print(f"[gauge] {entry['file']}", flush=True)
    per_subset = {s: pool(c) for s, c in clips.items()}
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "per_subset": per_subset, "clips": clips,
               "runtime_s": runtime}
    (args.out / "gauge.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, args.out / "gauge.md", sys.argv, runtime)
    print((args.out / "gauge.md").read_text())


if __name__ == "__main__":
    main()
