#!/usr/bin/env python
"""A3 headline: T/S/C split of the real error over a dump run; writes a3_tsc.{json,md}.

Per clip, design-1 rows go through :func:`l3d.tsc.tensor_from_table` and
:func:`l3d.tsc.two_way_split` for each error component; per subset, the per-cell parts
are summed over clips, so the shares are pooled. ``disagreement = 1 - T`` is the part
of the error that differs between queries of the same point (RQ1 asks whether it is
>= 20%). S share vs ``|t_s - t_t|`` is pooled the same way.

Example:
    .venv/bin/python scripts/l3d/run_a3_tsc.py
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

from l3d import tsc  # noqa: E402
from l3d.decompose import table_scale  # noqa: E402
from l3d.report import SUBSETS, fmt, iter_clips, jsonable, model_label  # noqa: E402

DEFAULT_DUMP = REPO / "output/l3d/dumps/OpenD4RT_48CLIP_9Mix_NoCropAUG_T48"
COMPONENTS = ("xyz", "log_depth", "image_plane")
PARTS = ("T", "S", "C", "SC")
LABEL = {"xyz": "3D error (camera t_t, / depth)", "log_depth": "depth (log ratio along ray)",
         "image_plane": "image-plane (perpendicular to ray)"}


def clip_parts(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Summed per-cell parts and S-vs-gap numerators/denominators of one clip."""
    scale = table_scale(d)
    out = {}
    for comp in COMPONENTS:
        tf = tsc.tensor_from_table(d, component=comp, scale=scale)
        sp = tsc.two_way_split(tf["E"], tf["mask"])
        gap = tsc.s_share_vs_gap(tf["E"], tf["mask"], tf["s_frames"], tf["t_frames"])
        num = np.nan_to_num(gap["s_effect_mean"] * gap["count"])
        with np.errstate(invalid="ignore", divide="ignore"):
            den = np.where(gap["share"] > 0, num / gap["share"], 0.0)
        out[comp] = {
            **{k: float(np.nansum(sp[k])) for k in PARTS + ("total",)},
            "n_cells": sp["n_cells"],
            "gap_num": num, "gap_den": np.nan_to_num(den), "gap_count": gap["count"], "gap_edges": gap["bin_edges"],
        }
    return out


def pool(clips: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_clips": len(clips)}
    for comp in COMPONENTS:
        tot = sum(c[comp]["total"] for c in clips)
        shares = {k: sum(c[comp][k] for c in clips) / tot if tot > 0 else float("nan") for k in PARTS}
        num = np.sum([c[comp]["gap_num"] for c in clips], axis=0)
        den = np.sum([c[comp]["gap_den"] for c in clips], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap_share = np.where(den > 0, num / den, np.nan)
        out[comp] = {
            "shares": shares,
            "disagreement": 1.0 - shares["T"],
            "n_cells": int(sum(c[comp]["n_cells"] for c in clips)),
            "mean_total": tot / max(sum(c[comp]["n_cells"] for c in clips), 1),
            "s_vs_gap": {"bin_edges": clips[0][comp]["gap_edges"], "share": gap_share,
                         "count": np.sum([c[comp]["gap_count"] for c in clips], axis=0)},
        }
    return out


def write_md(per_subset: dict[str, dict[str, Any]], path: Path, argv: list[str], runtime_s: float, label: str = "OpenD4RT 48CLIP") -> None:
    lines = [
        f"# A3: T/S/C split of the real error ({label}, design 1, clip [0, 48))",
        "",
        f"`{' '.join(argv)}` ({runtime_s:.0f} s). Per `(track, t_t)` cell, design 1 decodes 6 source frames x 6",
        "camera frames (`t_t` and 0 always among the cameras); the balanced two-way split gives the part shared by",
        "every query of the cell (T), the source and camera main effects (S, C) and their interaction (SC). Shares",
        "are sums over complete cells of all clips. Disagreement = 1 - T = the part that differs between queries of",
        "the same point.",
        "",
    ]
    for comp in COMPONENTS:
        lines += [f"## {LABEL[comp]}", "", "| Subset | Cells | T | S | C | SC | Disagreement |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s, r in per_subset.items():
            if r["n_clips"]:
                sh = r[comp]["shares"]
                lines.append(f"| `{s}` | {r[comp]['n_cells']} | {fmt(sh['T'])} | {fmt(sh['S'])} | {fmt(sh['C'])} | "
                             f"{fmt(sh['SC'])} | {fmt(r[comp]['disagreement'])} |")
        lines.append("")
    edges = next(iter(per_subset.values()))["log_depth"]["s_vs_gap"]["bin_edges"]
    heads = [f"[{int(a)}, {int(b)})" for a, b in zip(edges[:-1], edges[1:])]
    for comp in ("log_depth", "xyz"):
        lines += [f"## S share vs `|t_s - t_t|` ({LABEL[comp]})", "",
                  "| Subset | " + " | ".join(heads) + " |", "| --- | " + " | ".join("---:" for _ in heads) + " |"]
        for s, r in per_subset.items():
            if r["n_clips"]:
                g = r[comp]["s_vs_gap"]
                cells = [fmt(v) if n else "-" for v, n in zip(g["share"], g["count"])]
                lines.append(f"| `{s}` | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def write_png(per_subset: dict[str, dict[str, Any]], path: Path) -> None:
    """Stacked T / S / C / SC shares per subset, one panel per component."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a66", "#e4e3df"
    color = {"T": "#d6d5d0", "S": "#2a78d6", "C": "#eb6834", "SC": "#1baf7a"}  # T neutral; categorical slots 1-3
    name = {"T": "T (shared by every query)", "S": "S (source frame)", "C": "C (camera frame)", "SC": "S x C"}
    subsets = [s for s in per_subset if per_subset[s].get("n_clips")]
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(4.0 * len(COMPONENTS), 0.55 * len(subsets) + 1.6),
                             sharey=True, constrained_layout=True)
    for ax, comp in zip(axes, COMPONENTS):
        for y, s in enumerate(subsets):
            left = 0.0
            for part in PARTS:
                w = 100 * float(per_subset[s][comp]["shares"][part])
                ax.barh(y, w, left=left, height=0.62, color=color[part], edgecolor="white", linewidth=2)
                if w >= 7:
                    ax.text(left + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=7,
                            color=ink if part in ("T", "SC") else "white")
                left += w
        ax.axvline(80, color=muted, lw=1, ls=":", zorder=0)  # behind the bars: shows between them only
        ax.set_xlim(0, 100)
        ax.set_title(LABEL[comp], fontsize=9, color=ink, loc="left")
        ax.set_xlabel("share of mean squared error (%)", fontsize=8, color=ink)
        ax.tick_params(labelsize=7, colors=muted, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(subsets)))
    axes[0].set_yticklabels(subsets, fontsize=8, color=ink)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[p]) for p in PARTS]
    handles.append(plt.Line2D([], [], color=muted, lw=1, ls=":"))
    fig.legend(handles, [name[p] for p in PARTS] + ["T = 80%: 20% disagreement (RQ1 bar)"], loc="outside upper left",
               ncol=5, frameon=False, fontsize=8, labelcolor=ink)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--model-label", default=None, help="model name in the report (default: from the dump manifest)")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=0, help="clips per subset (0 = all)")
    ap.add_argument("--out", type=Path, default=REPO / "results/l3d/a3_tsc")
    ap.add_argument("--from-json", action="store_true", help="only redraw the figure from an existing a3_tsc.json")
    args = ap.parse_args()

    if args.from_json:
        write_png(json.loads((args.out / "a3_tsc.json").read_text())["per_subset"], args.out / "a3_tsc.png")
        return
    subsets = args.subsets.split(",")
    start = time.time()
    clips: dict[str, list[dict[str, Any]]] = {s: [] for s in subsets}
    for entry, d in iter_clips(args.dump, subsets, args.limit):
        clips[entry["subset"]].append(clip_parts(d))
        print(f"[a3_tsc] {entry['file']}", flush=True)
    per_subset = {s: pool(c) if c else {"n_clips": 0} for s, c in clips.items()}
    runtime = time.time() - start
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: str(v) for k, v in vars(args).items()}, "per_subset": per_subset, "runtime_s": runtime}
    (args.out / "a3_tsc.json").write_text(json.dumps(jsonable(payload), indent=1))
    write_md(per_subset, args.out / "a3_tsc.md", sys.argv, runtime, label=model_label(args.dump, args.model_label))
    write_png(jsonable(per_subset), args.out / "a3_tsc.png")
    print((args.out / "a3_tsc.md").read_text())


if __name__ == "__main__":
    main()
