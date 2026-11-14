# Matched-pair GBC comparison

Runs: 80; combinations: 40; max relative difference to the saved GBC: 5.5e-13.
Cells: mean over partitions; five-fold datasets add [min, max] over folds.

## A. Paper-faithful GBC (PCA 64, spherical, training only), paper form

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | -3.351 [-3.87, -2.758] | -10.11 [-10.82, -9.157] | -19.83 [-21.15, -18.13] | -12.08 [-12.88, -10.72] | -55.59 [-59.55, -52.03] |
| aquamonitor | -6.582 [-7.157, -6.277] | -7.989 [-8.712, -7.657] | -9.778 [-10.61, -9.031] | -12.47 [-13.1, -11.76] | -27.78 [-28.4, -26.73] |
| bioscan5m | -1063 | -1649 | -68.59 | -1546 | -5142 |
| df24 | -686.4 | -1046 | -69.41 | -3005 | -3.244e+04 |
| food101 | -2.556 | -4.555 | -12.53 | -2.122 | -763.5 |
| oxford_pets | -0.1549 | -0.1573 | -1.943 | -0.5705 | -5.558 |
| cifar100 | -5.484 | -3.684 | -12.68 | -15.08 | -88.25 |
| fgvc_aircraft | -92.64 | -645.7 | -252.1 | -21.15 | -1707 |

## A. Authors' code-release form

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | -3.562 [-4.148, -2.942] | -10.44 [-11.18, -9.463] | -22.15 [-23.81, -20.28] | -13.47 [-14.4, -11.87] | -62.21 [-67.03, -57.55] |
| aquamonitor | -6.979 [-7.514, -6.72] | -8.301 [-9.049, -8.023] | -10.32 [-11.14, -9.467] | -14.16 [-14.81, -13.27] | -33.15 [-34.2, -31.52] |
| bioscan5m | -1135 | -1806 | -73.42 | -1675 | -5652 |
| df24 | -712.7 | -1123 | -70.58 | -3013 | -3.357e+04 |
| food101 | -2.85 | -4.852 | -13 | -2.234 | -788.5 |
| oxford_pets | -0.1637 | -0.1933 | -2.039 | -0.5835 | -5.839 |
| cifar100 | -5.614 | -3.755 | -13.39 | -16.1 | -96.98 |
| fgvc_aircraft | -94.33 | -652.3 | -245.3 | -21.17 | -1838 |

## A. Spearman(BC paper form, BC code form)

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 0.998 [0.998, 0.998] | 0.999 [0.999, 0.999] | 0.999 [0.998, 0.999] | 0.997 [0.997, 0.998] | 0.996 [0.996, 0.997] |
| aquamonitor | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] | 0.999 [0.999, 0.999] | 0.999 [0.999, 0.999] | 0.988 [0.986, 0.990] |
| bioscan5m | 0.998 | 0.998 | 0.999 | 0.998 | 0.995 |
| df24 | 0.999 | 0.999 | 0.999 | 1.000 | 0.999 |
| food101 | 0.998 | 0.997 | 0.999 | 1.000 | 0.999 |
| oxford_pets | 0.997 | 0.995 | 1.000 | 0.999 | 0.999 |
| cifar100 | 0.997 | 0.999 | 0.996 | 0.998 | 0.998 |
| fgvc_aircraft | 0.993 | 0.993 | 0.999 | 0.999 | 0.993 |

## D. Spherical pca64: val/train ratio of mean BC over identical pairs

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 0.740 [0.534, 0.925] | 0.707 [0.581, 0.808] | 0.793 [0.685, 0.867] | 0.787 [0.597, 0.959] | 0.785 [0.675, 0.928] |
| aquamonitor | 0.641 [0.543, 0.740] | 0.661 [0.546, 0.827] | 0.687 [0.579, 0.832] | 0.757 [0.653, 0.938] | 0.716 [0.608, 0.866] |
| bioscan5m | 0.376 | 0.378 | 0.399 | 0.405 | 0.411 |
| df24 | 0.374 | 0.389 | 0.315 | 0.342 | 0.336 |
| food101 | 0.499 | 0.638 | 0.848 | 0.704 | 0.825 |
| oxford_pets | 0.615 | 1.058 | 0.536 | 0.342 | 0.398 |
| cifar100 | 0.810 | 0.778 | 0.727 | 0.702 | 0.756 |
| fgvc_aircraft | 1.009 | 1.093 | 1.002 | 1.048 | 1.072 |

## D. Spherical pca16: val/train ratio of mean BC over identical pairs

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 0.923 [0.880, 0.999] | 0.911 [0.859, 0.977] | 0.914 [0.870, 0.972] | 0.934 [0.840, 1.055] | 0.904 [0.835, 0.984] |
| aquamonitor | 0.880 [0.824, 0.988] | 0.914 [0.847, 1.049] | 0.923 [0.815, 1.062] | 0.921 [0.828, 1.015] | 0.841 [0.797, 0.886] |
| bioscan5m | 0.753 | 0.766 | 0.719 | 0.763 | 0.751 |
| df24 | 0.737 | 0.741 | 0.807 | 0.741 | 0.722 |
| food101 | 0.925 | 0.956 | 0.946 | 0.924 | 0.944 |
| oxford_pets | 0.795 | 0.565 | 0.842 | 0.810 | 0.852 |
| cifar100 | 0.944 | 0.970 | 0.917 | 0.970 | 0.918 |
| fgvc_aircraft | 0.985 | 1.037 | 1.022 | 0.994 | 1.032 |

## D. Spherical fda16: val/train ratio of mean BC over identical pairs

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 1.316 [1.081, 1.617] | 1.442 [1.309, 1.677] | 1.240 [1.184, 1.345] | 1.325 [1.206, 1.442] | 1.460 [1.255, 1.698] |
| aquamonitor | 0.942 [0.824, 1.112] | 0.939 [0.847, 1.056] | 0.945 [0.842, 1.054] | 0.956 [0.853, 1.176] | 1.004 [0.865, 1.146] |
| bioscan5m | 0.710 | 0.680 | 0.542 | 0.680 | 0.779 |
| df24 | 0.751 | 0.721 | 0.742 | 0.761 | 0.744 |
| food101 | 0.946 | 0.986 | 0.990 | 0.997 | 1.038 |
| oxford_pets | 2.627 | 1.102 | 1.781 | 1.522 | 1.261 |
| cifar100 | 0.998 | 1.007 | 0.961 | 0.895 | 0.956 |
| fgvc_aircraft | 1.496 | 2.203 | 1.753 | 1.597 | 2.381 |

## D. Spherical pca16: mean BC val - train over identical pairs

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | -3.17e-03 [-5.05e-03, -3.11e-05] | -7.37e-03 [-1.16e-02, -1.95e-03] | -1.22e-02 [-1.90e-02, -4.04e-03] | -6.92e-03 [-1.69e-02, 5.71e-03] | -2.30e-02 [-3.95e-02, -3.49e-03] |
| aquamonitor | -5.38e-03 [-7.91e-03, -5.26e-04] | -4.25e-03 [-7.43e-03, 2.35e-03] | -4.67e-03 [-1.13e-02, 3.39e-03] | -6.98e-03 [-1.53e-02, 1.27e-03] | -2.81e-02 [-3.59e-02, -2.02e-02] |
| bioscan5m | -6.86e-03 | -8.55e-03 | -9.78e-04 | -1.08e-02 | -3.53e-02 |
| df24 | -5.83e-03 | -5.81e-03 | -3.59e-04 | -1.46e-02 | -6.45e-02 |
| food101 | -2.70e-04 | -2.90e-04 | -2.17e-03 | -2.36e-03 | -1.96e-02 |
| oxford_pets | -4.19e-04 | -9.71e-04 | -2.82e-03 | -2.59e-03 | -7.01e-03 |
| cifar100 | -7.73e-04 | -4.32e-04 | -3.14e-03 | -1.26e-03 | -8.76e-03 |
| fgvc_aircraft | -4.85e-04 | 6.36e-03 | 2.54e-03 | -5.40e-05 | 1.49e-02 |

## D. Spherical fda16: mean BC val - train over identical pairs

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 1.92e-03 [5.15e-04, 3.40e-03] | 4.19e-03 [3.21e-03, 5.88e-03] | 3.21e-03 [2.53e-03, 4.29e-03] | 3.03e-03 [1.93e-03, 3.68e-03] | 5.77e-03 [3.25e-03, 8.16e-03] |
| aquamonitor | -7.73e-04 [-2.04e-03, 9.44e-04] | -9.93e-04 [-2.33e-03, 7.81e-04] | -1.03e-03 [-2.98e-03, 8.24e-04] | -7.57e-04 [-2.26e-03, 2.40e-03] | -4.20e-05 [-2.14e-03, 2.16e-03] |
| bioscan5m | -3.41e-03 | -4.97e-03 | -5.19e-03 | -3.90e-03 | -2.00e-03 |
| df24 | -2.76e-03 | -4.52e-03 | -1.08e-03 | -2.27e-03 | -6.72e-03 |
| food101 | -3.91e-04 | -1.43e-04 | -1.27e-04 | -7.50e-06 | 1.76e-03 |
| oxford_pets | 1.05e-03 | 1.99e-04 | 2.03e-03 | 6.63e-04 | 9.62e-04 |
| cifar100 | -1.63e-05 | 7.47e-05 | -4.33e-04 | -1.40e-03 | -8.99e-04 |
| fgvc_aircraft | 4.30e-03 | 1.62e-02 | 1.77e-02 | 1.91e-03 | 3.79e-02 |

## C. Spherical, d = 16, matched pairs: FDA - PCA mean BC (train)

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | -3.46e-02 [-3.64e-02, -3.35e-02] | -7.27e-02 [-7.60e-02, -6.86e-02] | -1.28e-01 [-1.32e-01, -1.23e-01] | -9.35e-02 [-9.70e-02, -9.10e-02] | -2.23e-01 [-2.28e-01, -2.14e-01] |
| aquamonitor | -3.39e-02 [-3.63e-02, -3.07e-02] | -3.33e-02 [-3.41e-02, -3.25e-02] | -4.17e-02 [-4.44e-02, -3.97e-02] | -7.21e-02 [-7.34e-02, -7.07e-02] | -1.62e-01 [-1.64e-01, -1.60e-01] |
| bioscan5m | -1.61e-02 | -2.09e-02 | 7.85e-03 | -3.33e-02 | -1.33e-01 |
| df24 | -1.11e-02 | -6.20e-03 | 2.34e-03 | -4.68e-02 | -2.06e-01 |
| food101 | 3.67e-03 | 3.62e-03 | -2.71e-02 | -2.84e-02 | -3.03e-01 |
| oxford_pets | -1.41e-03 | -2.82e-04 | -1.53e-02 | -1.24e-02 | -4.38e-02 |
| cifar100 | -6.21e-03 | -3.51e-03 | -2.65e-02 | -2.89e-02 | -8.66e-02 |
| fgvc_aircraft | -2.38e-02 | -1.60e-01 | -9.03e-02 | -6.06e-03 | -4.44e-01 |

## C. Spherical, d = 16, matched pairs: FDA - PCA mean BC (val)

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | -2.96e-02 [-3.30e-02, -2.83e-02] | -6.12e-02 [-6.85e-02, -5.76e-02] | -1.12e-01 [-1.23e-01, -1.05e-01] | -8.36e-02 [-9.52e-02, -7.75e-02] | -1.95e-01 [-2.09e-01, -1.83e-01] |
| aquamonitor | -2.93e-02 [-3.27e-02, -2.62e-02] | -3.01e-02 [-3.52e-02, -2.63e-02] | -3.80e-02 [-4.23e-02, -3.38e-02] | -6.59e-02 [-7.05e-02, -5.97e-02] | -1.34e-01 [-1.43e-01, -1.25e-01] |
| bioscan5m | -1.26e-02 | -1.74e-02 | 3.64e-03 | -2.64e-02 | -9.93e-02 |
| df24 | -8.02e-03 | -4.92e-03 | 1.61e-03 | -3.45e-02 | -1.48e-01 |
| food101 | 3.54e-03 | 3.76e-03 | -2.50e-02 | -2.60e-02 | -2.82e-01 |
| oxford_pets | 6.36e-05 | 8.88e-04 | -1.05e-02 | -9.11e-03 | -3.58e-02 |
| cifar100 | -5.45e-03 | -3.00e-03 | -2.38e-02 | -2.90e-02 | -7.87e-02 |
| fgvc_aircraft | -1.90e-02 | -1.50e-01 | -7.51e-02 | -4.09e-03 | -4.21e-01 |

## F. Spherical fda16, pairs with min val units 5-9: val/train mean BC

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 3.342 [0.662, 8.556] | 1.433 [0.843, 2.575] | 2.293 [0.844, 4.938] | 2.344 [0.738, 7.353] | 5.549 [0.885, 21.726] |
| aquamonitor | 0.916 [0.600, 1.269] | 0.922 [0.509, 1.233] | 0.962 [0.773, 1.215] | 0.978 [0.829, 1.358] | 0.995 [0.757, 1.245] |
| bioscan5m | 0.652 | 0.618 | 0.491 | 0.626 | 0.739 |
| df24 | 0.709 | 0.673 | 0.714 | 0.718 | 0.697 |
| food101 | n/a | n/a | n/a | n/a | n/a |
| oxford_pets | 23.281 | 1.837 | 2.474 | 3.900 | 1.584 |
| cifar100 | n/a | n/a | n/a | n/a | n/a |
| fgvc_aircraft | n/a | n/a | n/a | n/a | n/a |

## F. Spherical fda16, pairs with min val units >=20: val/train mean BC

| dataset | dinov3_l | dinov2_l | bioclip2 | siglip2 | ijepa_h14 |
|---|---|---|---|---|---|
| finbenthic2 | 1.445 [1.402, 1.467] | 1.577 [1.479, 1.649] | 1.302 [1.177, 1.431] | 1.441 [1.243, 1.675] | 1.521 [1.384, 1.667] |
| aquamonitor | 1.030 [1.020, 1.060] | 1.015 [0.954, 1.091] | 1.013 [0.956, 1.104] | 1.013 [0.923, 1.077] | 1.081 [0.982, 1.141] |
| bioscan5m | 0.874 | 0.832 | 0.655 | 0.844 | 0.919 |
| df24 | 0.885 | 0.879 | 0.864 | 0.909 | 0.903 |
| food101 | 0.946 | 0.986 | 0.990 | 0.997 | 1.038 |
| oxford_pets | n/a | n/a | n/a | n/a | n/a |
| cifar100 | 0.998 | 1.007 | 0.961 | 0.895 | 0.956 |
| fgvc_aircraft | 1.496 | 2.203 | 1.753 | 1.597 | 2.381 |

## Counts

- Pair sets of train and val differ (train-valid > common): spherical 0, diagonal 0, full 1070 of 1280 run x arm rows; full with no common pair: 175
- Full form, differing pair sets with both ratios defined: 917; previous ratio > 1 but matched ratio <= 1: 0; previous <= 1 but matched > 1: 33
- Spherical pca64: ratio > 1 in 6 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical pca8: ratio > 1 in 10 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical pca16: ratio > 1 in 7 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical pca32: ratio > 1 in 9 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical fda8: ratio > 1 in 50 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical fda16: ratio > 1 in 43 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical fda32: ratio > 1 in 39 of 80 runs; not reported (train mean BC < 1e-6) in 0
- Spherical d = 8, matched: 80 of 80 runs eligible; FDA mean BC train < PCA in 76, val < PCA in 74; FDA ratio > PCA ratio in 66
- Spherical d = 16, matched: 80 of 80 runs eligible; FDA mean BC train < PCA in 76, val < PCA in 74; FDA ratio > PCA ratio in 67
- Spherical d = 32, matched: 80 of 80 runs eligible; FDA mean BC train < PCA in 75, val < PCA in 75; FDA ratio > PCA ratio in 61
- Spherical fda16, runs with both strata: 60; val/train ratio higher for low-support pairs (5-9) than for >= 20 in 18
