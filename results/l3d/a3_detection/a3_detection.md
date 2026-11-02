# A3: detection by each check vs native confidence (OpenD4RT 48CLIP, clip [0, 48))

`scripts/l3d/run_a3_detection.py` (733 s). Rows of all clips of a subset are pooled. Error = `|s P - X| / |X|`
of the scored query (mean over the pair for source swap); target = the component Table 1 says the check
sees. Confidence is scored as `-sigmoid(conf)` on the same rows. AUROC: positives = top 20% of errors.
E-AURC: excess area under the risk-coverage curve over the oracle (lower is better). Gate 1 (ii) asks for
rho(residual, target) >= 0.3.

Label-free K: fx(fit) / fx(GT), median over clips: `adt_mini` 0.966, `po_mini` 1.027, `pstudio_mini` 1.016, `ds_mini` 1.000.

## source swap, model u2 (design 5 vs 1)

Target: S depth: |e_d(t1) - e_d(t2)|.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 3014160 | 0.587 | 0.149 | 0.404 | 0.014 | 0.805 | 0.542 | 0.0511 | 0.1345 | 0.266 | 0.158 |
| `po_mini` | 3068544 | 0.499 | 0.008 | 0.348 | 0.098 | 0.755 | 0.581 | 0.0259 | 0.0500 | 0.065 | 0.182 |
| `pstudio_mini` | 2550672 | 0.490 | 0.022 | 0.339 | 0.153 | 0.718 | 0.590 | 0.0179 | 0.0281 | 0.074 | 0.297 |
| `ds_mini` | 3072000 | 0.511 | 0.130 | 0.328 | 0.230 | 0.719 | 0.654 | 0.0282 | 0.0337 | 0.188 | 0.195 |

## source swap, GT u2 control (design 1)

Target: S depth: |e_d(t1) - e_d(t2)|.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 3014160 | 0.767 | 0.154 | 0.381 | 0.011 | 0.772 | 0.537 | 0.0530 | 0.1143 | 0.206 | 0.193 |
| `po_mini` | 3068544 | 0.806 | 0.010 | 0.290 | 0.107 | 0.680 | 0.592 | 0.0246 | 0.0340 | -0.028 | 0.209 |
| `pstudio_mini` | 2550672 | 0.787 | 0.018 | 0.229 | 0.130 | 0.623 | 0.565 | 0.0193 | 0.0216 | 0.044 | 0.308 |
| `ds_mini` | 3072000 | 0.851 | 0.135 | 0.242 | 0.216 | 0.643 | 0.648 | 0.0290 | 0.0286 | 0.095 | 0.154 |

## camera swap, SE(3) fit (design 1)

Target: non-rigid camera-frame error.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 968853 | 0.903 | 0.158 | 0.464 | -0.030 | 0.832 | 0.502 | 0.0465 | 0.1418 | 0.178 | 0.073 |
| `po_mini` | 867575 | 0.910 | -0.026 | 0.181 | 0.078 | 0.609 | 0.568 | 0.0297 | 0.0348 | -0.011 | 0.062 |
| `pstudio_mini` | 771176 | 0.823 | -0.043 | 0.207 | 0.052 | 0.639 | 0.520 | 0.0179 | 0.0232 | -0.021 | 0.043 |
| `ds_mini` | 947537 | 0.904 | 0.100 | 0.193 | 0.199 | 0.635 | 0.639 | 0.0295 | 0.0282 | 0.162 | 0.049 |

## camera swap, Sim(3) fit (design 1)

Target: non-rigid camera-frame error.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 968853 | 0.968 | 0.158 | 0.461 | -0.030 | 0.817 | 0.502 | 0.0508 | 0.1418 | 0.166 | 0.064 |
| `po_mini` | 867575 | 0.989 | -0.026 | 0.171 | 0.078 | 0.605 | 0.568 | 0.0303 | 0.0348 | -0.015 | 0.058 |
| `pstudio_mini` | 771176 | 0.996 | -0.043 | 0.137 | 0.052 | 0.597 | 0.520 | 0.0205 | 0.0232 | -0.038 | 0.037 |
| `ds_mini` | 947537 | 0.983 | 0.100 | 0.189 | 0.199 | 0.633 | 0.639 | 0.0297 | 0.0282 | 0.144 | 0.047 |

## ray center, K fit + R* (design 1, t_s = t_t)

Target: image-plane error of P(u, t, t, t).

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 400850 | 0.136 | -0.117 | 0.372 | 0.161 | 0.707 | 0.628 | 0.0476 | 0.0723 | 0.400 | 0.087 |
| `po_mini` | 295708 | 0.052 | 0.085 | 0.163 | 0.216 | 0.576 | 0.671 | 0.0297 | 0.0211 | 0.193 | 0.070 |
| `pstudio_mini` | 308609 | 0.096 | 0.182 | 0.171 | 0.343 | 0.601 | 0.719 | 0.0170 | 0.0124 | 0.220 | 0.063 |
| `ds_mini` | 450367 | 0.337 | 0.134 | 0.025 | 0.250 | 0.505 | 0.669 | 0.0389 | 0.0214 | 0.074 | 0.075 |

## ray center, GT K and pose (not label-free)

Target: image-plane error of P(u, t, t, t).

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 400850 | 0.507 | -0.117 | 0.517 | 0.161 | 0.680 | 0.628 | 0.0522 | 0.0723 | 0.019 | 0.392 |
| `po_mini` | 295708 | 0.462 | 0.085 | 0.405 | 0.216 | 0.686 | 0.671 | 0.0245 | 0.0211 | 0.156 | 0.125 |
| `pstudio_mini` | 308609 | 0.227 | 0.182 | 0.285 | 0.343 | 0.678 | 0.719 | 0.0142 | 0.0124 | 0.282 | 0.078 |
| `ds_mini` | 450367 | 0.413 | 0.134 | 0.051 | 0.250 | 0.509 | 0.669 | 0.0386 | 0.0214 | 0.112 | 0.088 |

## reprojection with K fit, px (analysis only)

Target: image-plane error.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 80906 | 0.313 | -0.124 | 0.429 | 0.089 | 0.724 | 0.552 | 0.0363 | 0.0618 | 0.383 | 34.656 |
| `po_mini` | 61622 | 0.173 | 0.076 | 0.147 | 0.167 | 0.559 | 0.643 | 0.0324 | 0.0215 | 0.200 | 61.815 |
| `pstudio_mini` | 64371 | 0.063 | 0.178 | 0.135 | 0.325 | 0.585 | 0.708 | 0.0180 | 0.0123 | 0.225 | 47.442 |
| `ds_mini` | 90120 | 0.615 | 0.132 | 0.071 | 0.249 | 0.544 | 0.669 | 0.0375 | 0.0215 | 0.182 | 70.750 |

## static rigidity, GT static mask (analysis only)

Target: depth error |e_d|.

| Subset | Rows | rho(res, target) | rho(conf, target) | rho(res, error) | rho(conf, error) | AUROC res | AUROC conf | E-AURC res | E-AURC conf | rho(res, conf) | median res / median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 1084512 | 0.223 | 0.070 | 0.187 | -0.078 | 0.588 | 0.433 | 0.0574 | 0.0900 | 0.216 | 0.032 |
| `po_mini` | 1193184 | 0.165 | 0.004 | 0.199 | -0.008 | 0.653 | 0.521 | 0.0299 | 0.0460 | -0.047 | 0.035 |
| `pstudio_mini` | 40608 | -0.033 | -0.033 | -0.016 | -0.060 | 0.532 | 0.559 | 0.0858 | 0.0924 | 0.085 | 0.025 |
| `ds_mini` | 1227648 | 0.234 | 0.259 | 0.240 | 0.262 | 0.687 | 0.678 | 0.0294 | 0.0322 | 0.208 | 0.029 |
