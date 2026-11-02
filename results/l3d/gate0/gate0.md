# Gate 0: OpenD4RT_48CLIP_9Mix_NoCropAUG on WorldTrack (PASS)

Run: 2026-11-01 on AWS g6e.xlarge (1x L40S), `bash scripts/l3d/run_gate0.sh`, which runs the unmodified upstream
`eval_track3d_in_worldtrack.py --num-frames 64 --query-chunk-size 4096` at commit a32207b. The run took 46.7 min.
The raw output is `output/gate0_48clip/summary.txt` (not committed). Pass rule: |APD global - README| <= 0.02 for each subset.

| Subset | APD global | README | Delta | EPE global (README) | APD dyn (README) | EPE dyn (README) | Queries (README) | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `adt_mini` | 0.7220 | 0.7220 | 0.0000 | 0.2758 (0.2758) | 0.7325 (0.7325) | 0.3198 (0.3199) | 22187 (22187) | yes |
| `po_mini` | 0.6799 | 0.6799 | 0.0000 | 0.3177 (0.3178) | 0.7425 (0.7425) | 0.2593 (0.2593) | 53468 (53468) | yes |
| `pstudio_mini` | 0.7959 | 0.7960 | -0.0001 | 0.1754 (0.1753) | 0.7959 (0.7960) | 0.1754 (0.1753) | 8720 (8720) | yes |
| `ds_mini` | 0.7248 | 0.7248 | 0.0000 | 0.2959 (0.2959) | 0.7488 (0.7488) | 0.2754 (0.2755) | 52462 (52462) | yes |

Notes:
- Upstream skips one adt sequence, `Apartment_release_decoration_skeleton_seq139_8`, because frame 0 has no
  visible queries. Its dump table has no design-0 rows.
- Separate adt-only and po-only reruns gave identical numbers, so the eval is deterministic on this box.
- `extrinsics_w2c` is present in adt (300x4x4), po and ds (128x4x4) and tum. It is absent in pstudio, whose camera is fixed.
