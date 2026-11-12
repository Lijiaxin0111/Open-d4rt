# WorldTrack baselines: OpenD4RT vs ARROW vs TrackEverything

Run 2026-11-11/12 on the AWS g6e.xlarge (1x L40S). Every number comes from `scripts/l3d/score_worldtrack_preds.py`, which applies the upstream Gate 0 metric code (`_metrics_for_sequence`, `_aggregate_results` of `eval_track3d_in_worldtrack.py`) to each model's saved predictions: first 64 frames, frame-0 queries, camera-0 coordinates, one global median scale per sequence, APD over {0.1, 0.3, 0.5, 1.0}, means over sequences. All three models are scored on the same query set (checked). Cells are **APD / EPE**; higher APD and lower EPE are better. Regenerate with `python scripts/l3d/run_baselines_table.py`.

## In-image queries (ARROW's rule)

Frame-0 visible queries whose pixel lies inside the image (`-0.5 <= u <= W - 0.5`, same for v); `score_worldtrack_preds.py --in-image`.

| Subset | OpenD4RT | ARROW | TrackEverything (Pi3) | Queries |
| --- | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.722 / 0.276 | 0.936 / 0.082 | 0.919 / 0.097 | 22168 (19 dropped) |
| `po_mini` | 0.680 / 0.318 | 0.853 / 0.153 | 0.756 / 0.395 | 53458 (10 dropped) |
| `pstudio_mini` | 0.796 / 0.175 | 0.858 / 0.133 | 0.639 / 0.312 | 8706 (14 dropped) |
| `ds_mini` | 0.767 / 0.243 | 0.880 / 0.143 | 0.762 / 0.251 | 45163 (7299 dropped) |
| **mean** | 0.741 / 0.253 | 0.881 / 0.128 | 0.769 / 0.264 |  |

## All queries (upstream Gate 0 rule)

Frame-0 visible queries with a finite pixel and depth, no image bound. OpenD4RT reproduces Gate 0 exactly.

| Subset | OpenD4RT | ARROW | TrackEverything (Pi3) | Queries |
| --- | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.722 / 0.276 | 0.936 / 0.082 | 0.919 / 0.097 | 22187 |
| `po_mini` | 0.680 / 0.318 | 0.853 / 0.153 | 0.756 / 0.395 | 53468 |
| `pstudio_mini` | 0.796 / 0.175 | 0.857 / 0.133 | 0.639 / 0.312 | 8720 |
| `ds_mini` | 0.725 / 0.296 | 0.820 / 0.209 | 0.695 / 0.696 | 52462 |
| **mean** | 0.731 / 0.266 | 0.867 / 0.144 | 0.752 / 0.375 |  |

## Dynamic points, in-image queries

Cells are **APD dyn / EPE dyn** (points whose total GT motion exceeds 0.01; scale refit on them).

| Subset | OpenD4RT | ARROW | TrackEverything (Pi3) | Queries |
| --- | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.732 / 0.320 | 0.906 / 0.094 | 0.880 / 0.113 | 22168 (19 dropped) |
| `po_mini` | 0.742 / 0.259 | 0.870 / 0.138 | 0.769 / 0.358 | 53458 (10 dropped) |
| `pstudio_mini` | 0.796 / 0.175 | 0.858 / 0.133 | 0.639 / 0.312 | 8706 (14 dropped) |
| `ds_mini` | 0.821 / 0.184 | 0.915 / 0.093 | 0.703 / 0.344 | 45163 (7299 dropped) |
| **mean** | 0.773 / 0.235 | 0.887 / 0.115 | 0.748 / 0.282 |  |

## Check against ARROW's own evaluation

| Subset | ARROW official APD | Ours, in-image | ARROW official EPE | Ours, in-image |
| --- | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.9356 | 0.9356 | 0.0818 | 0.0818 |
| `po_mini` | 0.8526 | 0.8526 | 0.1532 | 0.1532 |
| `pstudio_mini` | 0.8575 | 0.8575 | 0.1333 | 0.1333 |
| `ds_mini` | 0.8800 | 0.8800 | 0.1426 | 0.1426 |

Largest APD difference: 1.4e-05. ARROW's official 4-subset mean APD is 0.8814, as in its paper (0.8814).

## Notes

- **OpenD4RT**: `OpenD4RT_48CLIP_9Mix_NoCropAUG` through the unmodified upstream eval, with `scripts/l3d/run_opend4rt_worldtrack_preds.py` saving the predictions it scores. The rerun's upstream summary is identical to Gate 0 (`results/l3d/gate0/`).
- **ARROW** (github.com/ifradlin/arrow at e0e1a76, weights `ifradlin/arrow`): inputs exactly as its own WorldTrack task builds them (max side 504), queried with our set by `scripts/l3d/run_arrow_worldtrack_ourqueries.py`.
- **TrackEverything** (github.com/ayushjain1144/trackeverything at dfb29b5, weights `katefgroup/TrackEverything`, sparse mode) via `scripts/l3d/run_trackeverything_worldtrack.py`, with **Pi3** geometry: the default VGGT-Omega checkpoint is a private cluster path. These numbers probably understate the paper's setup.
- **The query rule only matters on ds**: 13.9% of its queries project off-image (some at |u| ~ 1e6 px, from near-zero depth); adt, po and pstudio lose 19, 10 and 14 queries. Dropping them raises ds APD by 0.04-0.07 for all three models.
- ARROW's paper lists OpenD4RT at 0.7378 mean APD; we get 0.7306 with all queries and 0.7412 in-image. The query rule explains part of the gap but overshoots; the rest is unexplained (possibly how ARROW ran OpenD4RT).
- ARROW's own EPE dyn divides by every sequence (a sequence with no dynamic points counts as 0); ours averages only over sequences that have dynamic points, so ARROW's own figure is lower. Compare EPE dyn within one scorer only.
- Every pstudio point is dynamic, so its dyn numbers equal the overall ones. One adt sequence (`Apartment_release_decoration_skeleton_seq139_8`) has no frame-0 queries; 199 sequences are scored.
- Mean model seconds per sequence (inference only, L40S): ARROW 3.8 s; TrackEverything (Pi3) 37.1 s; OpenD4RT's whole upstream eval took 46 min for 199 sequences (about 14 s each, including loading). TrackEverything's Pi3 ran on a slow PyTorch RoPE2D fallback.
- Raw predictions and per-sequence scores: `output/l3d/baselines/` (not committed).
