#!/usr/bin/env python
"""A1: GT error decomposition over a dump run; writes a1.{json,md}.

Per clip :func:`l3d.decompose.from_table`; per subset, shares are pooled over clips
(each clip weighted by its number of rows times its mean squared error) and the
per-pair camera-frame numbers are medians over all ``(t, c)`` pairs.

Example:
    .venv/bin/python scripts/l3d/run_a1_decompose.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from l3d import decompose  # noqa: E402
from l3d.report import SUBSETS, jsonable  # noqa: E402
from l3d.table import load_table  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
PAIR_KEYS = ("gauge_rot_deg", "gauge_trans_rel", "gauge_log_scale", "total_rms_rel", "nonrigid_rms_rel")


def clip_summary(d: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Scalars for one clip, plus its per-pair camera-frame arrays (pooled per subset later)."""
    r = decompose.from_table(d)
    split, cam, sf = r["depth_split"], r["camera_frame"], r["self"]
    pairs = {k: np.asarray(cam.get(k, np.zeros(0)), dtype=np.float64) for k in PAIR_KEYS}
    n_pts = np.asarray(cam.get("n_points", np.zeros(0)), dtype=np.float64)
    out = {
        "scale": r["scale"],
        "n_groups": int(np.unique(r["labels"][r["labels"] >= 0]).size),
        "self": sf,
        "depth_shares": split["shares"],
        "depth_ms_total": split["ms_total"],
        "camera_nonrigid_share": cam.get("nonrigid_share", float("nan")),
        "camera_n_pairs": int(n_pts.size),
        "camera_tot2": float((n_pts * pairs["total_rms_rel"] ** 2).sum()),
        "camera_nr2": float((n_pts * pairs["nonrigid_rms_rel"] ** 2).sum()),
    }
    return out, pairs


def _pool(clips: list[dict[str, Any]], pairs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    ok = [c for c in clips if "error" not in c]
    if not ok:
        return {"n_clips": 0}
    w_self = np.array([c["self"]["n"] * c["self"]["sq_mean"] for c in ok])
    radial = np.array([c["self"]["radial_share"] for c in ok])
    w_depth = np.array([c["self"]["n"] * c["depth_ms_total"] for c in ok])
    depth = {k: float(np.nansum(w_depth * np.array([c["depth_shares"][k] for c in ok])) / np.nansum(w_depth))
             for k in ok[0]["depth_shares"]}
    tot2 = sum(c["camera_tot2"] for c in ok)
    nr2 = sum(c["camera_nr2"] for c in ok)
    cat = {k: np.concatenate([p[k] for p in pairs]) if pairs else np.zeros(0) for k in PAIR_KEYS}
    med = lambda key: float(np.nanmedian(np.abs(cat[key]))) if cat[key].size else float("nan")  # noqa: E731
    return {
        "n_clips": len(ok),
        "self_n": int(sum(c["self"]["n"] for c in ok)),
        "radial_share": float(np.nansum(w_self * radial) / np.nansum(w_self)),
        "perp_share": float(np.nansum(w_self * (1 - radial)) / np.nansum(w_self)),
        "e_d_rms": float(np.sqrt(np.nansum(np.array([c["self"]["n"] * c["self"]["e_d_rms"] ** 2 for c in ok]))
                                 / sum(c["self"]["n"] for c in ok))),
        "e_d_mean_median_clip": float(np.nanmedian([c["self"]["e_d_mean"] for c in ok])),
        "px_median_median_clip": float(np.nanmedian([c["self"]["px_median"] for c in ok])),
        "depth_shares": depth,
        "camera_nonrigid_share": nr2 / tot2 if tot2 > 0 else float("nan"),
        "camera_gauge_share": 1 - nr2 / tot2 if tot2 > 0 else float("nan"),
        "camera_n_pairs": int(cat["total_rms_rel"].size),
        "pair_median": {k: med(k) for k in PAIR_KEYS},
    }


def write_md(per_subset: dict[str, dict[str, Any]], path: Path, argv: list[str], runtime_s: float,
             errors: list[str]) -> None:
    f = lambda x, p=3: "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"  # noqa: E731
    lines = [
        "# A1: GT error decomposition (OpenD4RT 48CLIP, clip [0, 48))",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s). Shares are pooled over clips; pair numbers are medians over all",
        "`(t, c)` pairs (absolute values). Scale is each clip's global median scale (design 5 excluded).",
        "",
        "## Self-frame error (`t_c == t_t`, design 0)",
        "",
        "| Subset | Clips | Rows | Depth (radial) share | Image-plane share | e_d RMS | median clip e_d mean | median px |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, r in per_subset.items():
        if not r.get("n_clips"):
            continue
        lines.append(f"| `{s}` | {r['n_clips']} | {r['self_n']} | {f(r['radial_share'])} | {f(r['perp_share'])} | "
                     f"{f(r['e_d_rms'])} | {f(r['e_d_mean_median_clip'])} | {f(r['px_median_median_clip'], 1)} |")
    lines += [
        "",
        "## Depth error split (per motion group: offset = time mean, jitter = per-frame group mean, residual = per point)",
        "",
        "| Subset | Offset | Jitter | Residual |",
        "| --- | ---: | ---: | ---: |",
    ]
    for s, r in per_subset.items():
        if r.get("n_clips"):
            ds = r["depth_shares"]
            lines.append(f"| `{s}` | {f(ds['offset'])} | {f(ds['jitter'])} | {f(ds['residual'])} |")
    lines += [
        "",
        "## Camera-frame error (cross-frame vs self-frame mapped by GT pose; designs 0 and 1)",
        "",
        "| Subset | Pairs | Sim(3) gauge share | Non-rigid share | rot (deg) | trans / depth | abs log s | total RMS / depth | non-rigid RMS / depth |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s, r in per_subset.items():
        if r.get("n_clips"):
            pm = r["pair_median"]
            lines.append(f"| `{s}` | {r['camera_n_pairs']} | {f(r['camera_gauge_share'])} | {f(r['camera_nonrigid_share'])} | "
                         f"{f(pm['gauge_rot_deg'], 2)} | {f(pm['gauge_trans_rel'], 4)} | {f(pm['gauge_log_scale'], 4)} | "
                         f"{f(pm['total_rms_rel'], 4)} | {f(pm['nonrigid_rms_rel'], 4)} |")
    if errors:
        lines += ["", "## Skipped clips", ""] + [f"- {e}" for e in errors]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--out", type=Path, default=REPO / "results/l3d/a1_decompose")
    args = ap.parse_args()

    manifest = json.loads((args.dump / "manifest.json").read_text())
    subsets = args.subsets.split(",")
    start = time.time()
    clips: dict[str, list[dict[str, Any]]] = {s: [] for s in subsets}
    pairs: dict[str, list[dict[str, np.ndarray]]] = {s: [] for s in subsets}
    errors: list[str] = []
    for entry in manifest["clips"]:
        s = entry["subset"]
        if s not in clips or entry.get("status") != "ok" or (args.limit and len(clips[s]) >= args.limit):
            continue
        row: dict[str, Any] = {"subset": s, "sequence": entry["sequence"], "file": entry["file"]}
        try:
            summary, p = clip_summary(load_table(args.dump / entry["file"]))
            row.update(summary)
            pairs[s].append(p)
        except Exception as exc:  # noqa: BLE001 - a clip without self-frame rows should not stop the run
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"`{entry['file']}`: {row['error']}")
        clips[s].append(row)
        print(f"[a1] {entry['file']} {'ERROR ' + row['error'] if 'error' in row else 'ok'}", flush=True)

    per_subset = {s: _pool(clips[s], pairs[s]) for s in subsets}
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "per_subset": per_subset, "clips": clips,
               "runtime_s": runtime}
    (args.out / "a1.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, args.out / "a1.md", sys.argv, runtime, errors)
    print((args.out / "a1.md").read_text())


if __name__ == "__main__":
    main()
