# Part A by visibility, motion and parallax (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_splits.py` (64 s; code `l3d/splits.py`). Sums are pooled over each subset's clips. Columns:

- **Rows**: design-0 rows (frame-0 queries, every target frame) in the bin, and their share of the subset.
- **APD / EPE**: tracking accuracy of those rows decoded in camera 0 (the upstream protocol inside the clip), with the clip's global median scale; pooled over rows, so not the upstream per-sequence mean.
- **Rel. err / depth share**: self-frame (`t_c = t_t`) error |P - X| / |X| and the share of its squared error along the ray (A1's depth share).
- **Cells, T / S / C / SC**: complete design-1 cells (one point and target frame, crossed source x camera frames) in the bin and the T/S/C shares of their 3D error (A3); `1 - T` is the disagreement.
- Parallax of a design-0 row uses `t_s = 0`, so it grows with the frame gap; a design-1 cell uses its sources' median. pstudio has a fixed camera, so all of it is in the first parallax bin.

## Visibility (GT, at the target frame)

| Subset | Bin | Rows | Share | APD | EPE | Rel. err | Depth share | Cells | T | S | C | SC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | visible | 952365 | 0.89 | 0.776 | 0.220 | 0.114 | 0.81 | 81063 | 0.62 | 0.22 | 0.17 | 0.00 |
| `adt_mini` | occluded | 112611 | 0.11 | 0.572 | 0.409 | 0.309 | 0.79 | 20041 | 0.94 | 0.01 | 0.05 | 0.00 |
| `po_mini` | visible | 1753550 | 0.68 | 0.685 | 0.311 | 0.058 | 0.94 | 61661 | 0.67 | 0.31 | 0.01 | 0.00 |
| `po_mini` | occluded | 812914 | 0.32 | 0.646 | 0.376 | 0.073 | 0.92 | 40739 | 0.84 | 0.14 | 0.01 | 0.00 |
| `pstudio_mini` | visible | 363443 | 0.87 | 0.840 | 0.144 | 0.048 | 0.65 | 64496 | 0.74 | 0.24 | 0.02 | 0.00 |
| `pstudio_mini` | occluded | 55117 | 0.13 | 0.761 | 0.201 | 0.068 | 0.73 | 21040 | 0.77 | 0.22 | 0.01 | 0.00 |
| `ds_mini` | visible | 2364906 | 0.94 | 0.713 | 0.313 | 0.089 | 0.85 | 91818 | 0.91 | 0.09 | 0.00 | 0.00 |
| `ds_mini` | occluded | 153270 | 0.06 | 0.652 | 0.456 | 0.135 | 0.79 | 10582 | 0.87 | 0.13 | 0.00 | 0.00 |

## Motion (GT static: world displacement over the clip < 1% of median depth)

| Subset | Bin | Rows | Share | APD | EPE | Rel. err | Depth share | Cells | T | S | C | SC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | static | 832992 | 0.78 | 0.792 | 0.212 | 0.090 | 0.82 | 80984 | 0.94 | 0.01 | 0.05 | 0.00 |
| `adt_mini` | dynamic | 231984 | 0.22 | 0.618 | 0.338 | 0.291 | 0.66 | 20120 | 0.87 | 0.06 | 0.06 | 0.00 |
| `po_mini` | static | 1338288 | 0.52 | 0.615 | 0.391 | 0.065 | 0.94 | 55000 | 0.90 | 0.02 | 0.08 | 0.00 |
| `po_mini` | dynamic | 1228176 | 0.48 | 0.736 | 0.266 | 0.061 | 0.91 | 47400 | 0.83 | 0.17 | 0.00 | 0.00 |
| `pstudio_mini` | static | 19536 | 0.05 | 0.445 | 0.554 | 0.166 | 0.95 | 3104 | 0.93 | 0.07 | 0.00 | 0.00 |
| `pstudio_mini` | dynamic | 399024 | 0.95 | 0.848 | 0.132 | 0.045 | 0.45 | 82432 | 0.71 | 0.27 | 0.02 | 0.00 |
| `ds_mini` | static | 1882560 | 0.75 | 0.697 | 0.329 | 0.078 | 0.96 | 71360 | 0.93 | 0.07 | 0.00 | 0.00 |
| `ds_mini` | dynamic | 635616 | 0.25 | 0.748 | 0.300 | 0.135 | 0.46 | 31040 | 0.77 | 0.22 | 0.00 | 0.00 |

## Parallax (angle at the GT point between the source and target cameras, degrees)

| Subset | Bin | Rows | Share | APD | EPE | Rel. err | Depth share | Cells | T | S | C | SC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | [0, 1) | 229163 | 0.22 | 0.719 | 0.291 | 0.109 | 0.88 | 26880 | 0.89 | 0.08 | 0.03 | 0.00 |
| `adt_mini` | [1, 5) | 354710 | 0.33 | 0.757 | 0.237 | 0.128 | 0.82 | 46409 | 0.98 | 0.00 | 0.02 | 0.00 |
| `adt_mini` | >= 5 | 481103 | 0.45 | 0.768 | 0.218 | 0.151 | 0.62 | 27815 | 0.90 | 0.02 | 0.08 | 0.00 |
| `po_mini` | [0, 1) | 1521278 | 0.59 | 0.661 | 0.366 | 0.064 | 0.96 | 63785 | 0.74 | 0.25 | 0.01 | 0.00 |
| `po_mini` | [1, 5) | 613549 | 0.24 | 0.684 | 0.287 | 0.060 | 0.81 | 27443 | 0.85 | 0.14 | 0.01 | 0.00 |
| `po_mini` | >= 5 | 431637 | 0.17 | 0.698 | 0.273 | 0.064 | 0.77 | 11172 | 0.71 | 0.28 | 0.01 | 0.00 |
| `pstudio_mini` | [0, 1) | 418560 | 1.00 | 0.829 | 0.151 | 0.051 | 0.66 | 85536 | 0.75 | 0.23 | 0.01 | 0.00 |
| `pstudio_mini` | [1, 5) | 0 | | | | | | 0 | | | | |
| `pstudio_mini` | >= 5 | 0 | | | | | | 0 | | | | |
| `ds_mini` | [0, 1) | 1994437 | 0.79 | 0.703 | 0.326 | 0.081 | 0.89 | 92437 | 0.91 | 0.09 | 0.00 | 0.00 |
| `ds_mini` | [1, 5) | 511049 | 0.20 | 0.735 | 0.299 | 0.130 | 0.67 | 9908 | 0.88 | 0.12 | 0.00 | 0.00 |
| `ds_mini` | >= 5 | 12690 | 0.01 | 0.692 | 0.461 | 0.375 | 0.40 | 55 | 0.63 | 0.36 | 0.01 | 0.00 |
