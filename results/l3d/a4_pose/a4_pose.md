# A4: pose baselines from the decoded grid (OpenD4RT 48CLIP, design 2, clip [0, 48))

`scripts/l3d/run_a4_pose.py` (548 s). Per grid pair `(t, c)`: a 32x32 pixel grid of frame t decoded in
cameras t and c (`c` in {0, t-4, t+4}), outputs scaled by the clip scale, `T_{t->c}` fitted by each method and
compared with the GT relative pose. Translation errors are relative to the clip's median GT depth. ATE / RPE:
trajectory `T_{t->0}` vs GT after a Sim(3) alignment, median over clips (not computed for static cameras).

Label-free K on the grid: fx(fit) / fx(GT), median over clips: `adt_mini` 0.970, `po_mini` 1.025, `pstudio_mini` 1.020, `ds_mini` 1.001.

## Relative pose per pair (median over pairs)

| Subset | Pairs | GT rot (deg) | GT trans / depth | Method | rot err (deg) | trans err / depth | beats Umeyama (trans) | ATE / depth | RPE trans / depth | RPE rot (deg) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 6850 | 3.62 | 0.0488 | Umeyama | 0.504 | 0.0156 | n/a | 0.0105 | 0.0048 | 0.207 |
| | | | | Umeyama, conf-weighted | 0.504 | 0.0156 | 0.47 | 0.0105 | 0.0048 | 0.207 |
| | | | | ray center (K fit) | 0.504 | 0.0236 | 0.28 | 0.0128 | 0.0081 | 0.207 |
| | | | | ray center (GT K) | 0.504 | 0.1435 | 0.04 | 0.0268 | 0.0173 | 0.207 |
| `po_mini` | 6850 | 1.11 | 0.0078 | Umeyama | 0.299 | 0.0061 | n/a | 0.0073 | 0.0040 | 0.180 |
| | | | | Umeyama, conf-weighted | 0.299 | 0.0061 | 0.52 | 0.0073 | 0.0040 | 0.180 |
| | | | | ray center (K fit) | 0.299 | 0.0082 | 0.35 | 0.0080 | 0.0047 | 0.180 |
| | | | | ray center (GT K) | 0.299 | 0.0673 | 0.10 | 0.0155 | 0.0106 | 0.180 |
| `pstudio_mini` | 6850 | 0.00 | 0.0000 | Umeyama | 0.164 | 0.0048 | n/a | n/a | n/a | n/a |
| | | | | Umeyama, conf-weighted | 0.164 | 0.0048 | 0.53 | n/a | n/a | n/a |
| | | | | ray center (K fit) | 0.164 | 0.0083 | 0.23 | n/a | n/a | n/a |
| | | | | ray center (GT K) | 0.164 | 0.1503 | 0.00 | n/a | n/a | n/a |
| `ds_mini` | 6850 | 0.22 | 0.0034 | Umeyama | 0.119 | 0.0017 | n/a | 0.0016 | 0.0010 | 0.051 |
| | | | | Umeyama, conf-weighted | 0.119 | 0.0017 | 0.54 | 0.0016 | 0.0010 | 0.051 |
| | | | | ray center (K fit) | 0.119 | 0.0032 | 0.16 | 0.0018 | 0.0013 | 0.051 |
| | | | | ray center (GT K) | 0.119 | 0.0041 | 0.11 | 0.0020 | 0.0014 | 0.051 |

## Label-free camera residuals vs pose error (Gate 1 (iv): median residual >= 20% of the median pose error)

| Subset | median camera-swap residual (SE(3)) | median ray-center spread | median Umeyama trans err / depth | camera swap / pose err | ray center / pose err |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 0.0034 | 0.0069 | 0.0156 | 0.22 | 0.44 |
| `po_mini` | 0.0019 | 0.0037 | 0.0061 | 0.32 | 0.61 |
| `pstudio_mini` | 0.0011 | 0.0031 | 0.0048 | 0.24 | 0.64 |
| `ds_mini` | 0.0013 | 0.0030 | 0.0017 | 0.78 | 1.79 |
