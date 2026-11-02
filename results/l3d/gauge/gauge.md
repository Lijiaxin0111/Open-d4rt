# GAUGE baseline (GT anchors; a reference, not label-free), OpenD4RT 48CLIP, clip [0, 48)

`scripts/l3d/run_gauge.py` (316 s). World-frame outputs (design 0, `t_c = 0`) scaled by the clip's global
median scale; GAUGE (`l3d/gauge.py`: per-group per-frame ray scale about the model's camera centers + median
translation) and a per-group Sim(3) from the same anchors. Groups = GT motion groups (the paper's oracle
variant); anchors = whole GT tracks. Metrics on non-anchor points, upstream APD thresholds, no re-alignment,
mean over clips. Dynamic = total GT motion > 0.01 (upstream rule).

| Subset | Anchors | Points | APD base | APD GAUGE | APD Sim(3)/group | EPE base | EPE GAUGE | EPE Sim(3)/group | EPE gain GAUGE | Paper EPE gain (DQS / Full) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `adt_mini` | 1% (4/clip) | all | 0.727 | 0.725 | 0.644 | 0.2634 | 0.3319 | 0.6477 | -26.0% | +24.7% / +29.5% |
| `adt_mini` | 1% (4/clip) | dyn | 0.676 | 0.730 | 0.705 | 0.2867 | 0.2826 | 0.2727 | 1.4% | +24.7% / +29.5% |
| `adt_mini` | 5% (22/clip) | all | 0.726 | 0.775 | 0.786 | 0.2643 | 0.2205 | 0.2293 | 16.6% | +24.7% / +29.5% |
| `adt_mini` | 5% (22/clip) | dyn | 0.676 | 0.828 | 0.810 | 0.2870 | 0.1819 | 0.1862 | 36.6% | +24.7% / +29.5% |
| `po_mini` | 1% (10/clip) | all | 0.688 | 0.691 | 0.600 | 0.3079 | 0.3490 | 0.6392 | -13.3% | +43.4% / +18.2% |
| `po_mini` | 1% (10/clip) | dyn | 0.754 | 0.776 | 0.729 | 0.2476 | 0.2340 | 0.4124 | 5.5% | +43.4% / +18.2% |
| `po_mini` | 5% (53/clip) | all | 0.687 | 0.753 | 0.744 | 0.3088 | 0.2605 | 0.2671 | 15.6% | +43.4% / +18.2% |
| `po_mini` | 5% (53/clip) | dyn | 0.754 | 0.803 | 0.780 | 0.2481 | 0.2153 | 0.2324 | 13.2% | +43.4% / +18.2% |
| `pstudio_mini` | 1% (1/clip) | all | 0.807 | 0.869 | 0.820 | 0.1658 | 0.1277 | 0.1771 | 22.9% | +51.8% / +25.6% |
| `pstudio_mini` | 1% (1/clip) | dyn | 0.807 | 0.869 | 0.820 | 0.1658 | 0.1277 | 0.1771 | 22.9% | +51.8% / +25.6% |
| `pstudio_mini` | 5% (8/clip) | all | 0.807 | 0.920 | 0.903 | 0.1660 | 0.0925 | 0.1087 | 44.3% | +51.8% / +25.6% |
| `pstudio_mini` | 5% (8/clip) | dyn | 0.807 | 0.920 | 0.903 | 0.1660 | 0.0925 | 0.1087 | 44.3% | +51.8% / +25.6% |
| `ds_mini` | 1% (10/clip) | all | 0.728 | 0.720 | 0.650 | 0.2940 | 0.3002 | 0.4284 | -2.1% | +31.9% / -8.3% |
| `ds_mini` | 1% (10/clip) | dyn | 0.751 | 0.784 | 0.672 | 0.2882 | 0.2296 | 0.4933 | 20.4% | +31.9% / -8.3% |
| `ds_mini` | 5% (52/clip) | all | 0.728 | 0.732 | 0.734 | 0.2936 | 0.3249 | 0.2736 | -10.7% | +31.9% / -8.3% |
| `ds_mini` | 5% (52/clip) | dyn | 0.752 | 0.823 | 0.778 | 0.2874 | 0.1968 | 0.2366 | 31.5% | +31.9% / -8.3% |

Paper column: GAUGE Table A1 for D4RT (their unsupervised groups, 5% anchors, full sequences); for
orientation only.
