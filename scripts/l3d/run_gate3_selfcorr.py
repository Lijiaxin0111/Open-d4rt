#!/usr/bin/env python
"""Gate 3 input: 2D error of the model's self-correspondences; writes selfcorr.{json,md,png}.

A self-correspondence is the design-4 output ``u2 = uv_hat(u1, t1, t2, t2)``, the pixel
source swap uses for its second query. Error = ``|p_uv - g_uv|`` in pixels of the model's
256x256 input (normalized coordinates x 255), over rows with ``t1 != t2``. Gate 3: use
self-correspondences if the median is <= 2 px, otherwise CoTracker3. Reported for rows
visible at ``t2`` by GT and, as training would filter them, by the model's own visibility
(``sigmoid(vis) > 0.5``).

Example:
    .venv/bin/python scripts/l3d/run_gate3_selfcorr.py
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

from l3d.report import SUBSETS, fmt, iter_clips, jsonable  # noqa: E402
from l3d.table import select  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
GATE_PX = 2.0
GAP_EDGES = np.array([1, 2, 4, 8, 16, 32, 48])
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4


def clip_rows(d: dict[str, Any]) -> dict[str, np.ndarray]:
    r = select(d, design="selfmatch")
    keep = (r["q_ts"] != r["q_tt"]) & np.isfinite(r["p_uv"]).all(-1) & np.isfinite(r["g_uv"]).all(-1)
    du = (r["p_uv"][keep] - r["g_uv"][keep]).astype(np.float64)
    h, w = (int(x) for x in d["hw_native"])
    return {
        "err256": np.linalg.norm(du * 255.0, axis=-1),
        "err_native": np.linalg.norm(du * np.array([w - 1, h - 1]), axis=-1),
        "gt_vis": r["g_vis"][keep].astype(bool),
        "model_vis": r["p_vis_logit"][keep] > 0.0,
        "gap": np.abs(r["q_tt"][keep].astype(np.int64) - r["q_ts"][keep].astype(np.int64)),
    }


def _stats(e: np.ndarray) -> dict[str, float]:
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "median": float(np.median(e)), "mean": float(np.mean(e)),
            "p90": float(np.quantile(e, 0.9)), "frac_le_gate": float(np.mean(e <= GATE_PX))}


def summarize(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    gt, mv = rows["gt_vis"], rows["model_vis"]
    e = rows["err256"]
    which = np.digitize(rows["gap"], GAP_EDGES) - 1
    gap_median = [float(np.median(e[gt & (which == i)])) if np.any(gt & (which == i)) else float("nan")
                  for i in range(GAP_EDGES.size - 1)]
    return {
        "gt_visible": _stats(e[gt]),
        "model_visible": _stats(e[mv]),
        "native_px_gt_visible": _stats(rows["err_native"][gt]),
        "gate3_pass": bool(np.median(e[gt]) <= GATE_PX) if gt.any() else None,
        "gate3_pass_model_vis": bool(np.median(e[mv]) <= GATE_PX) if mv.any() else None,
        "vis_recall": float(np.mean(mv[gt])) if gt.any() else float("nan"),
        "gap_edges": GAP_EDGES,
        "gap_median": gap_median,
    }


def write_md(per_subset: dict[str, dict[str, Any]], path: Path, argv: list[str], runtime_s: float) -> None:
    lines = [
        "# Gate 3 input: self-correspondence 2D error (OpenD4RT 48CLIP, design 4, clip [0, 48))",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s). Error of the model's own `u2 = uv_hat(u1, t1, t2, t2)` against GT, in",
        "pixels of the 256x256 model input, rows with `t1 != t2`. Gate 3 uses self-correspondences if the median is",
        f"<= {GATE_PX:g} px, otherwise CoTracker3. GT-visible: visible at `t2` by GT. Model-visible: `sigmoid(vis) > 0.5`",
        "at `t2`, the label-free filter training would use. Design 4 queries only frames where the track is GT-visible,",
        "so every row is GT-visible and \"kept by model vis.\" is the share the model's filter would keep.",
        "",
        "| Subset | Rows (GT-visible) | Median px | Mean px | p90 px | <= 2 px | Gate 3 | Median px (model-visible) "
        "| Gate 3 (model-visible) | Kept by model vis. | Median native px |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    yn = lambda b: "n/a" if b is None else ("**pass**" if b else "fail (use CoTracker3)")  # noqa: E731
    for s, r in per_subset.items():
        g, m = r["gt_visible"], r["model_visible"]
        lines.append(
            f"| `{s}` | {g['n']} | {fmt(g.get('median'), 2)} | {fmt(g.get('mean'), 2)} | {fmt(g.get('p90'), 2)} | "
            f"{fmt(g.get('frac_le_gate'), 2)} | {yn(r['gate3_pass'])} | {fmt(m.get('median'), 2)} | "
            f"{yn(r['gate3_pass_model_vis'])} | {fmt(r['vis_recall'], 2)} | "
            f"{fmt(r['native_px_gt_visible'].get('median'), 2)} |"
        )
    edges = GAP_EDGES
    heads = [f"[{a}, {b})" for a, b in zip(edges[:-1], edges[1:])]
    lines += ["", "## Median px (GT-visible) by frame gap `|t2 - t1|`", "",
              "| Subset | " + " | ".join(heads) + " |", "| --- | " + " | ".join("---:" for _ in heads) + " |"]
    for s, r in per_subset.items():
        lines.append(f"| `{s}` | " + " | ".join(fmt(v, 2) for v in r["gap_median"]) + " |")
    lines += ["", "`selfcorr.png`: share of GT-visible self-correspondences within each error (the table above is its",
              "text view)."]
    path.write_text("\n".join(lines) + "\n")


def write_png(curves: dict[str, np.ndarray], path: Path) -> None:
    """ECDF of the GT-visible 2D error per subset, log x, with the Gate 3 bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e4e3df"
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.axvline(GATE_PX, color=muted, lw=1, ls="--", zorder=1)
    ax.text(GATE_PX * 1.06, 0.03, "Gate 3 bar (2 px)", fontsize=7, color=muted, va="bottom")
    ax.axhline(0.5, color=grid, lw=0.8, zorder=0)
    for (s, e), col in zip(curves.items(), COLORS):
        x = np.sort(e)
        y = np.arange(1, x.size + 1) / x.size
        step = max(x.size // 2000, 1)
        med = float(np.median(x))
        ax.plot(x[::step], y[::step], color=col, lw=2, zorder=3, label=f"{s}  median {med:.2f} px")
        ax.plot([med], [0.5], marker="o", ms=8, color=col, mec="white", mew=2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ticks = [0.1, 0.3, 1, 2, 3, 10, 30, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("self-correspondence error (px at 256x256, log scale)", fontsize=8, color=ink)
    ax.set_ylabel("share of correspondences within error", fontsize=8, color=ink)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(labelsize=7, colors=muted, length=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper left", title="dot = median",
              title_fontsize=7)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--out", type=Path, default=REPO / "results/l3d/gate3_selfcorr")
    args = ap.parse_args()

    subsets = args.subsets.split(",")
    start = time.time()
    acc: dict[str, list[dict[str, np.ndarray]]] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        acc[entry["subset"]].append(clip_rows(d))
    pooled = {s: {k: np.concatenate([c[k] for c in v]) for k in v[0]} for s, v in acc.items() if v}
    per_subset = {s: summarize(r) for s, r in pooled.items()}
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "gate_px": GATE_PX, "per_subset": per_subset,
               "runtime_s": runtime}
    (args.out / "selfcorr.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, args.out / "selfcorr.md", sys.argv, runtime)
    write_png({s: r["err256"][r["gt_vis"]] for s, r in pooled.items()}, args.out / "selfcorr.png")
    print((args.out / "selfcorr.md").read_text())


if __name__ == "__main__":
    main()
