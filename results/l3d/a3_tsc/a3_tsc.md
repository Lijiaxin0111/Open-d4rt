# A3: T/S/C split of the real error (OpenD4RT 48CLIP, design 1, clip [0, 48))

`scripts/l3d/run_a3_tsc.py` (384 s). Per `(track, t_t)` cell, design 1 decodes 6 source frames x 6
camera frames (`t_t` and 0 always among the cameras); the balanced two-way split gives the part shared by
every query of the cell (T), the source and camera main effects (S, C) and their interaction (SC). Shares
are sums over complete cells of all clips. Disagreement = 1 - T = the part that differs between queries of
the same point.

## 3D error (camera t_t, / depth)

| Subset | Cells | T | S | C | SC | Disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 101104 | 0.940 | 0.011 | 0.049 | 0.000 | 0.060 |
| `po_mini` | 102400 | 0.840 | 0.145 | 0.013 | 0.001 | 0.160 |
| `pstudio_mini` | 85536 | 0.751 | 0.234 | 0.014 | 0.001 | 0.249 |
| `ds_mini` | 102400 | 0.904 | 0.094 | 0.001 | 0.001 | 0.096 |

## depth (log ratio along ray)

| Subset | Cells | T | S | C | SC | Disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 99549 | 0.769 | 0.033 | 0.186 | 0.012 | 0.231 |
| `po_mini` | 102399 | 0.637 | 0.342 | 0.017 | 0.004 | 0.363 |
| `pstudio_mini` | 85536 | 0.798 | 0.184 | 0.017 | 0.001 | 0.202 |
| `ds_mini` | 102400 | 0.883 | 0.116 | 0.001 | 0.001 | 0.117 |

## image-plane (perpendicular to ray)

| Subset | Cells | T | S | C | SC | Disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 101104 | 0.530 | 0.101 | 0.346 | 0.023 | 0.470 |
| `po_mini` | 102400 | 0.784 | 0.150 | 0.052 | 0.014 | 0.216 |
| `pstudio_mini` | 85536 | 0.855 | 0.127 | 0.017 | 0.001 | 0.145 |
| `ds_mini` | 102400 | 0.664 | 0.297 | 0.036 | 0.003 | 0.336 |

## S share vs `|t_s - t_t|` (depth (log ratio along ray))

| Subset | [0, 1) | [1, 2) | [2, 4) | [4, 8) | [8, 16) | [16, 32) | [32, 64) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.056 | 0.040 | 0.046 | 0.048 | 0.043 | 0.038 | 0.024 |
| `po_mini` | 0.339 | 0.315 | 0.313 | 0.332 | 0.343 | 0.353 | 0.354 |
| `pstudio_mini` | 0.169 | 0.153 | 0.201 | 0.177 | 0.204 | 0.193 | 0.165 |
| `ds_mini` | 0.093 | 0.095 | 0.104 | 0.112 | 0.116 | 0.122 | 0.134 |

## S share vs `|t_s - t_t|` (3D error (camera t_t, / depth))

| Subset | [0, 1) | [1, 2) | [2, 4) | [4, 8) | [8, 16) | [16, 32) | [32, 64) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.242 | 0.188 | 0.142 | 0.222 | 0.149 | 0.009 | 0.021 |
| `po_mini` | 0.419 | 0.250 | 0.245 | 0.386 | 0.354 | 0.065 | 0.137 |
| `pstudio_mini` | 0.239 | 0.218 | 0.280 | 0.246 | 0.267 | 0.227 | 0.182 |
| `ds_mini` | 0.091 | 0.082 | 0.075 | 0.094 | 0.085 | 0.099 | 0.113 |
