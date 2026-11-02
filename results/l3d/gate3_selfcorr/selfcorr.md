# Gate 3 input: self-correspondence 2D error (OpenD4RT 48CLIP, design 4, clip [0, 48))

`scripts/l3d/run_gate3_selfcorr.py` (18 s). Error of the model's own `u2 = uv_hat(u1, t1, t2, t2)` against GT, in
pixels of the 256x256 model input, rows with `t1 != t2`. Gate 3 uses self-correspondences if the median is
<= 2 px, otherwise CoTracker3. GT-visible: visible at `t2` by GT. Model-visible: `sigmoid(vis) > 0.5`
at `t2`, the label-free filter training would use. Design 4 queries only frames where the track is GT-visible,
so every row is GT-visible and "kept by model vis." is the share the model's filter would keep.

| Subset | Rows (GT-visible) | Median px | Mean px | p90 px | <= 2 px | Gate 3 | Median px (model-visible) | Gate 3 (model-visible) | Kept by model vis. | Median native px |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| `adt_mini` | 473089 | 1.98 | 3.54 | 6.85 | 0.50 | **pass** | 1.91 | **pass** | 0.95 | 3.97 |
| `po_mini` | 473021 | 1.90 | 3.20 | 6.40 | 0.52 | **pass** | 1.81 | **pass** | 0.85 | 5.49 |
| `pstudio_mini` | 402068 | 2.44 | 3.72 | 7.37 | 0.41 | fail (use CoTracker3) | 2.39 | fail (use CoTracker3) | 0.96 | 4.65 |
| `ds_mini` | 496247 | 0.97 | 1.57 | 2.99 | 0.82 | **pass** | 0.97 | **pass** | 1.00 | 3.69 |

## Median px (GT-visible) by frame gap `|t2 - t1|`

| Subset | [1, 2) | [2, 4) | [4, 8) | [8, 16) | [16, 32) | [32, 48) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 1.17 | 1.47 | 1.72 | 1.96 | 2.23 | 2.91 |
| `po_mini` | 1.08 | 1.30 | 1.58 | 1.84 | 2.26 | 2.75 |
| `pstudio_mini` | 1.06 | 1.37 | 1.88 | 2.56 | 3.26 | 4.09 |
| `ds_mini` | 0.68 | 0.75 | 0.86 | 0.93 | 1.08 | 1.36 |

`selfcorr.png`: share of GT-visible self-correspondences within each error (the table above is its
text view).
