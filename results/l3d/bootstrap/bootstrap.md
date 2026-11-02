# Bootstrap 95% CIs of the gate numbers (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_bootstrap.py` (693 s, 1000 resamples of each subset's clips, one per sequence). Point = full-data value of the
gate tables (bootstrap row statistics use a per-clip row subsample). Verdict: the whole CI on one side of the
threshold (pass = `>=` threshold, except Gate 3: `<=` 2 px). RQ1 rows: >= 20% disagreement, >= 50% offset.

| Subset | Statistic | Threshold | Point | 95% CI | Verdict |
| --- | --- | ---: | ---: | --- | --- |
| `adt_mini` | source_swap (ii) rho | >= 0.3 | 0.587 | [0.517, 0.661] | robust pass |
| `adt_mini` | camera_swap (ii) rho | >= 0.3 | 0.903 | [0.880, 0.923] | robust pass |
| `adt_mini` | ray_center (ii) rho | >= 0.3 | 0.136 | [-0.010, 0.267] | robust fail |
| `adt_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.045 | [0.025, 0.086] | robust fail |
| `adt_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.021 | [0.012, 0.034] | robust fail |
| `adt_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.196 | [0.124, 0.308] | **uncertain** |
| `adt_mini` | source_swap (iv) median ratio | >= 0.2 | 0.158 | [0.121, 0.204] | **uncertain** |
| `adt_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.220 | [0.155, 0.289] | **uncertain** |
| `adt_mini` | ray_center (iv) median ratio | >= 0.2 | 0.441 | [0.302, 0.594] | robust pass |
| `adt_mini` | Gate 3 median px | <= 2 | 1.982 | [1.799, 2.347] | **uncertain** |
| `adt_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.060 | [0.022, 0.177] | robust fail |
| `adt_mini` | RQ1 depth offset share | >= 0.5 | 0.440 | [0.305, 0.695] | **uncertain** |
| `adt_mini` | GAUGE 5% EPE gain %, all |  | 16.574 | [5.641, 26.716] |  |
| `adt_mini` | GAUGE 5% EPE gain %, dyn |  | 36.620 | [17.870, 55.199] |  |
| `po_mini` | source_swap (ii) rho | >= 0.3 | 0.499 | [0.466, 0.529] | robust pass |
| `po_mini` | camera_swap (ii) rho | >= 0.3 | 0.910 | [0.896, 0.926] | robust pass |
| `po_mini` | ray_center (ii) rho | >= 0.3 | 0.052 | [-0.020, 0.127] | robust fail |
| `po_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.347 | [0.291, 0.405] | robust pass |
| `po_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.273 | [0.180, 0.437] | **uncertain** |
| `po_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.072 | [0.046, 0.112] | robust fail |
| `po_mini` | source_swap (iv) median ratio | >= 0.2 | 0.182 | [0.164, 0.207] | **uncertain** |
| `po_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.316 | [0.238, 0.416] | robust pass |
| `po_mini` | ray_center (iv) median ratio | >= 0.2 | 0.605 | [0.443, 0.803] | robust pass |
| `po_mini` | Gate 3 median px | <= 2 | 1.899 | [1.755, 2.111] | **uncertain** |
| `po_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.160 | [0.116, 0.345] | **uncertain** |
| `po_mini` | RQ1 depth offset share | >= 0.5 | 0.104 | [0.082, 0.126] | robust fail |
| `po_mini` | GAUGE 5% EPE gain %, all |  | 15.631 | [10.457, 20.828] |  |
| `po_mini` | GAUGE 5% EPE gain %, dyn |  | 13.202 | [8.003, 18.608] |  |
| `pstudio_mini` | source_swap (ii) rho | >= 0.3 | 0.490 | [0.470, 0.514] | robust pass |
| `pstudio_mini` | camera_swap (ii) rho | >= 0.3 | 0.823 | [0.728, 0.832] | robust pass |
| `pstudio_mini` | ray_center (ii) rho | >= 0.3 | 0.096 | [-0.020, 0.207] | robust fail |
| `pstudio_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.185 | [0.113, 0.270] | **uncertain** |
| `pstudio_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.125 | [0.101, 0.166] | robust fail |
| `pstudio_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.337 | [0.244, 0.459] | robust pass |
| `pstudio_mini` | source_swap (iv) median ratio | >= 0.2 | 0.297 | [0.252, 0.315] | robust pass |
| `pstudio_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.236 | [0.212, 0.261] | robust pass |
| `pstudio_mini` | ray_center (iv) median ratio | >= 0.2 | 0.644 | [0.579, 0.718] | robust pass |
| `pstudio_mini` | Gate 3 median px | <= 2 | 2.440 | [2.231, 2.620] | robust fail |
| `pstudio_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.249 | [0.131, 0.361] | **uncertain** |
| `pstudio_mini` | RQ1 depth offset share | >= 0.5 | 0.653 | [0.530, 0.749] | robust pass |
| `pstudio_mini` | GAUGE 5% EPE gain %, all |  | 44.311 | [37.934, 49.747] |  |
| `pstudio_mini` | GAUGE 5% EPE gain %, dyn |  | 44.311 | [37.934, 49.747] |  |
| `ds_mini` | source_swap (ii) rho | >= 0.3 | 0.511 | [0.491, 0.533] | robust pass |
| `ds_mini` | camera_swap (ii) rho | >= 0.3 | 0.904 | [0.893, 0.915] | robust pass |
| `ds_mini` | ray_center (ii) rho | >= 0.3 | 0.337 | [0.298, 0.369] | **uncertain** |
| `ds_mini` | source_swap (iii) S+SC share of depth | >= 0.2 | 0.116 | [0.090, 0.148] | robust fail |
| `ds_mini` | camera_swap (iii) non-rigid share | >= 0.2 | 0.678 | [0.594, 0.741] | robust pass |
| `ds_mini` | ray_center (iii) image-plane share | >= 0.2 | 0.158 | [0.068, 0.298] | **uncertain** |
| `ds_mini` | source_swap (iv) median ratio | >= 0.2 | 0.195 | [0.180, 0.218] | **uncertain** |
| `ds_mini` | camera_swap (iv) median ratio | >= 0.2 | 0.784 | [0.715, 0.863] | robust pass |
| `ds_mini` | ray_center (iv) median ratio | >= 0.2 | 1.791 | [1.619, 1.977] | robust pass |
| `ds_mini` | Gate 3 median px | <= 2 | 0.973 | [0.927, 1.020] | robust pass |
| `ds_mini` | RQ1 disagreement (1 - T), 3D | >= 0.2 | 0.096 | [0.075, 0.124] | robust fail |
| `ds_mini` | RQ1 depth offset share | >= 0.5 | 0.166 | [0.087, 0.276] | robust fail |
| `ds_mini` | GAUGE 5% EPE gain %, all |  | -10.669 | [-42.206, 12.429] |  |
| `ds_mini` | GAUGE 5% EPE gain %, dyn |  | 31.507 | [18.780, 40.803] |  |
