## FIN-Benthic2

Per fold (coupled minus LW, held-out log-likelihood per val specimen):

 fold  pcs  classes  macro_mean  macro_ci_low  macro_ci_high  micro_mean  micro_ci_low  micro_ci_high  classes_coupled_better  classes_ci_above_0  classes_ci_below_0  numerical_failures  four_class_check_max_diff
fold0   16       36     -0.0055       -0.0871         0.0909     -0.0307       -0.0519        -0.0073                      10                   2                   9                   0                        0.0
fold0   64       36      0.7632        0.3380         1.3376      0.1084       -0.0093         0.2376                      18                   5                   4                   0                        0.0
fold1   16       35     -0.1231       -0.1676        -0.0827     -0.0568       -0.0770        -0.0366                       8                   0                   7                   0                        0.0
fold1   64       35      0.1900       -0.0254         0.3920     -0.1113       -0.1987        -0.0254                      14                   5                   7                   0                        0.0
fold2   16       34     -0.0716       -0.1175        -0.0152     -0.0529       -0.0736        -0.0295                       8                   1                   8                   0                        0.0
fold2   64       34      0.0169       -0.1440         0.1755     -0.1106       -0.2063        -0.0153                      14                   1                   5                   0                        0.0
fold3   16       35     -0.0621       -0.1087        -0.0120     -0.0392       -0.0613        -0.0168                      10                   0                   9                   0                        0.0
fold3   64       35      0.4098        0.1554         0.6755     -0.0530       -0.1630         0.0555                      17                   5                   7                   0                        0.0
fold4   16       35     -0.0188       -0.0729         0.0380     -0.0376       -0.0597        -0.0149                      13                   0                   6                   0                        0.0
fold4   64       35      0.4044        0.1796         0.6241     -0.0535       -0.1552         0.0466                      15                   5                   6                   0                        0.0

Across folds (mean, min, max):

    macro_mean                 micro_mean                
          mean     min     max       mean     min     max
pcs                                                      
16     -0.0562 -0.1231 -0.0055    -0.0434 -0.0568 -0.0307
64      0.3569  0.0169  0.7632    -0.0440 -0.1113  0.1084

By train-size stratum:

 pcs stratum  class_folds  taxa  class_diff_mean  class_diff_median  class_folds_coupled_better  fold_macro_mean  fold_macro_min  fold_macro_max  folds_ci_above_0  folds_ci_below_0
  16   15-63           25     7          -0.0776            -0.1707                           8          -0.1203         -0.5101          0.2122                 0                 1
  16  64-255          119    26          -0.0559            -0.0415                          32          -0.0557         -0.0825         -0.0402                 0                 4
  16   >=256           31     7          -0.0381            -0.0336                           9          -0.0383         -0.0557         -0.0267                 0                 4
  64   15-63           25     7           2.9660             3.0118                          20           2.9194          0.7421          4.7426                 4                 0
  64  64-255          119    26          -0.0410            -0.0572                          55          -0.0412         -0.0919         -0.0110                 0                 0
  64   >=256           31     7          -0.1959            -0.1823                           3          -0.1926         -0.2956         -0.0022                 0                 4

Parameters:

 pcs stratum  class_folds  coupled_alpha_keep_median  coupled_shrink_to_identity_median  lw_alpha_shrink_median  coupled_beta_median  share_beta_0  share_beta_1  share_alpha_1  lw_condition_median  coupled_condition_median
  16   15-63           25                     0.9241                             0.0759                  0.1723               0.7801           0.0        0.0000         0.1600              22.4374                   24.0349
  16  64-255          119                     0.9830                             0.0170                  0.0507               0.9343           0.0        0.0252         0.2269              49.1607                   50.3958
  16   >=256           31                     0.9956                             0.0044                  0.0262               0.9473           0.0        0.0000         0.1935              56.3057                   56.4415
  64   15-63           25                     0.8988                             0.1012                  0.2333               0.7249           0.0        0.0000         0.1200              60.4622                   82.5892
  64  64-255          119                     0.9792                             0.0208                  0.0685               0.9092           0.0        0.0336         0.1933             175.3007                  246.1081
  64   >=256           31                     0.9950                             0.0050                  0.0392               0.9289           0.0        0.0000         0.4516             259.0145                  301.1637

Corners (diagnostic; minus LW):

 pcs stratum  class_folds  class_scm_defined  coupled_minus_lw_mean  pooled_scm_minus_lw_mean  class_scm_minus_lw_mean  class_identity_minus_lw_mean  coupled_beats_lw  pooled_scm_beats_lw  class_scm_beats_lw  class_identity_beats_lw
  16   15-63           25                 21                -0.0776                   -3.1219                  -8.2816                       -5.2431                 8                    1                   1                        0
  16  64-255          119                119                -0.0559                   -4.0860                  -0.1639                       -5.8174                32                    0                  25                        0
  16   >=256           31                 31                -0.0381                   -3.5460                  -0.0270                       -6.3142                 9                    0                  13                        0
  64   15-63           25                  0                 2.9660                   -4.7664                      NaN                      -19.7131                20                    5                   0                        0
  64  64-255          119                119                -0.0410                  -16.2170                -259.9811                      -29.4362                55                    0                   0                        0
  64   >=256           31                 31                -0.1959                  -14.0226                  -1.9620                      -32.5965                 3                    0                   0                        0

LW reference vs lw_fit_diagnostics:

{
 "class_folds_here": 350,
 "class_folds_reference": 350,
 "matched": 350,
 "only_here": 0,
 "only_reference": 0,
 "max_abs_diff_val_loglik_lw": 0.0,
 "max_abs_diff_alpha_lw": 0.0,
 "n_train_equal": true,
 "n_val_equal": true
}

Examples:

 pcs            why        taxon_group    diff  n_train   beta  folds  lw_shrink  coupled_shrink  diff_min  diff_max
  16   largest_gain          Sialis_sp  1.6410     15.5 0.8205    2.0     0.4383          0.4883    0.7103    2.5718
  16   largest_loss      Leuctra_nigra -0.7863     15.0 0.6487    3.0     0.3506          0.2706   -1.4284   -0.3297
  16 smallest_class      Leuctra_nigra -0.7863     15.0 0.6487    3.0     0.3506          0.2706   -1.4284   -0.3297
  16  largest_class         Simuliidae -0.0491    564.0 0.9830    5.0     0.0153          0.0137   -0.0821    0.0153
  16   most_pooling      Leuctra_nigra -0.7863     15.0 0.6487    3.0     0.3506          0.2706   -1.4284   -0.3297
  64   largest_gain          Sialis_sp 10.5694     15.5 0.6572    2.0     0.4664          0.4284    5.5501   15.5887
  64   largest_loss Lepidostoma_hirtum -1.3605    170.0 0.9352    5.0     0.0495          0.0038   -2.0646   -0.8727
  64 smallest_class      Leuctra_nigra  3.8481     15.0 0.4568    3.0     0.4875          0.2581    3.0118    4.8384
  64  largest_class         Simuliidae -0.2136    564.0 0.9793    5.0     0.0216          0.0156   -0.4920    0.4346
  64   most_pooling      Leuctra_nigra  3.8481     15.0 0.4568    3.0     0.4875          0.2581    3.0118    4.8384

## AquaMonitor

Per fold (coupled minus LW, held-out log-likelihood per val specimen):

 fold  pcs  classes  macro_mean  macro_ci_low  macro_ci_high  micro_mean  micro_ci_low  micro_ci_high  classes_coupled_better  classes_ci_above_0  classes_ci_below_0  numerical_failures  four_class_check_max_diff
fold0   16       41     -0.1639       -0.2022        -0.1233     -0.0279       -0.0344        -0.0210                       6                   0                  18                   0                        0.0
fold0   64       41     -0.1561       -0.3633         0.0456     -0.0622       -0.0937        -0.0285                      11                   3                  14                   0                        0.0
fold1   16       41     -0.1239       -0.1808        -0.0630     -0.0249       -0.0317        -0.0170                      12                   0                  16                   0                        0.0
fold1   64       41     -0.0222       -0.2952         0.2624     -0.0647       -0.0985        -0.0263                      15                   3                  13                   0                        0.0
fold2   16       40     -0.1671       -0.2070        -0.1244     -0.0266       -0.0335        -0.0194                       7                   0                  13                   0                        0.0
fold2   64       40     -0.0142       -0.2614         0.2472     -0.0423       -0.0766        -0.0063                      19                   3                   6                   0                        0.0
fold3   16       41     -0.1735       -0.2209        -0.1247     -0.0262       -0.0342        -0.0172                       8                   0                  16                   0                        0.0
fold3   64       41      0.0975       -0.1712         0.3962     -0.0263       -0.0618         0.0125                      16                   2                   8                   0                        0.0
fold4   16       41     -0.1115       -0.1750        -0.0427     -0.0180       -0.0273        -0.0073                      10                   0                  18                   0                        0.0
fold4   64       41      0.3465        0.0262         0.6871      0.0064       -0.0442         0.0620                      18                   4                   8                   0                        0.0

Across folds (mean, min, max):

    macro_mean                 micro_mean                
          mean     min     max       mean     min     max
pcs                                                      
16     -0.1480 -0.1735 -0.1115    -0.0247 -0.0279 -0.0180
64      0.0503 -0.1561  0.3465    -0.0378 -0.0647  0.0064

By train-size stratum:

 pcs stratum  class_folds  taxa  class_diff_mean  class_diff_median  class_folds_coupled_better  fold_macro_mean  fold_macro_min  fold_macro_max  folds_ci_above_0  folds_ci_below_0
  16   15-63           39     9          -0.2616            -0.2245                          10          -0.2648         -0.3913         -0.1641                 0                 3
  16  64-255           95    19          -0.1888            -0.1635                          13          -0.1888         -0.2551         -0.1479                 0                 5
  16   >=256           70    14          -0.0291            -0.0042                          20          -0.0291         -0.0356         -0.0228                 0                 5
  64   15-63           39     9           1.0955             0.5477                          28           1.0761          0.3191          1.6955                 4                 0
  64  64-255           95    19          -0.2660            -0.1856                          36          -0.2660         -0.6316          0.1350                 0                 2
  64   >=256           70    14          -0.1018            -0.0220                          15          -0.1018         -0.1372         -0.0812                 0                 2

Parameters:

 pcs stratum  class_folds  coupled_alpha_keep_median  coupled_shrink_to_identity_median  lw_alpha_shrink_median  coupled_beta_median  share_beta_0  share_beta_1  share_alpha_1  lw_condition_median  coupled_condition_median
  16   15-63           39                     0.9192                             0.0808                  0.1211               0.8519           0.0           0.0         0.2308              48.7265                   29.9425
  16  64-255           95                     0.9640                             0.0360                  0.0567               0.9339           0.0           0.0         0.2105              52.9571                   41.7987
  16   >=256           70                     0.9978                             0.0022                  0.0062               0.9921           0.0           0.0         0.2143              90.5640                   74.0414
  64   15-63           39                     0.9418                             0.0582                  0.1536               0.8151           0.0           0.0         0.1282             126.2894                  145.4083
  64  64-255           95                     0.9649                             0.0351                  0.0727               0.8987           0.0           0.0         0.2632             224.9354                  264.8852
  64   >=256           70                     0.9980                             0.0020                  0.0081               0.9906           0.0           0.0         0.1286             417.9627                  405.8665

Corners (diagnostic; minus LW):

 pcs stratum  class_folds  class_scm_defined  coupled_minus_lw_mean  pooled_scm_minus_lw_mean  class_scm_minus_lw_mean  class_identity_minus_lw_mean  coupled_beats_lw  pooled_scm_beats_lw  class_scm_beats_lw  class_identity_beats_lw
  16   15-63           39                 39                -0.2616                   -5.0160                  -3.3039                       -6.4706                10                    1                   4                        0
  16  64-255           95                 95                -0.1888                   -4.6293                  -0.2482                       -6.3405                13                    0                  27                        0
  16   >=256           70                 70                -0.0291                   -4.7985                  -0.0058                       -6.9042                20                    0                  34                        0
  64   15-63           39                  0                 1.0955                  -15.8030                      NaN                      -30.0567                28                    0                   0                        0
  64  64-255           95                 95                -0.2660                  -20.2609                 -60.0371                      -33.5209                36                    0                   0                        0
  64   >=256           70                 70                -0.1018                  -22.2758                  -0.7037                      -40.3595                15                    0                   7                        0

LW reference vs lw_fit_diagnostics:

{
 "class_folds_here": 408,
 "class_folds_reference": 408,
 "matched": 408,
 "only_here": 0,
 "only_reference": 0,
 "max_abs_diff_val_loglik_lw": 0.0,
 "max_abs_diff_alpha_lw": 0.0,
 "n_train_equal": true,
 "n_val_equal": true
}

Examples:

 pcs            why            taxon_group    diff  n_train   beta  folds  lw_shrink  coupled_shrink  diff_min  diff_max
  16   largest_gain        Cloeon dipterum  0.0172    103.0 0.9339    5.0     0.1070          0.0767   -0.0888    0.1230
  16   largest_loss                Pedicia -0.6602     40.0 0.7620    5.0     0.1046          0.0095   -1.1889    0.0013
  16 smallest_class                 Sisyra -0.0169     35.0 0.7828    3.0     0.3093          0.3187   -0.7626    1.2490
  16  largest_class           Chironomidae  0.0000   7416.0 0.9985    5.0     0.0006          0.0001   -0.0013    0.0013
  16   most_pooling             Mystacides -0.1542     37.5 0.7299    4.0     0.2168          0.0730   -0.4435    0.0480
  64   largest_gain             Mystacides  3.1210     37.5 0.7071    4.0     0.2241          0.0379    1.3314    4.7117
  64   largest_loss Heptagenia dalecarlica -3.1039    137.0 0.8987    5.0     0.0615          0.0000   -4.6348   -2.3376
  64 smallest_class                 Sisyra  2.1951     35.0 0.7320    3.0     0.3535          0.2782   -0.2366    6.6050
  64  largest_class           Chironomidae  0.0013   7416.0 0.9980    5.0     0.0009          0.0002   -0.0046    0.0090
  64   most_pooling                Pedicia  2.1073     40.0 0.6856    5.0     0.1445          0.0256    0.0075    4.7026

