## FIN-Benthic2

Across folds (fold mean, min, max; within-space contrasts):

                                     macro_mean                         micro_mean                        
                                           mean         min         max       mean         min         max
space      contrast                                                                                       
pca64      coupled_vs_lw                 0.3569      0.0169      0.7632    -0.0440     -0.1113      0.1084
           normalisation                 0.0527      0.0304      0.0792     0.0181      0.0122      0.0223
           pooling_at_lw_shrink          0.5289      0.2345      0.7849     0.2213      0.0887      0.3021
           pooling_at_coupled_shrink    12.7558      4.9888     35.2428     5.0530      3.1162      9.6325
           shrink_at_no_pooling        -12.4902    -34.8924     -4.8119    -5.1241     -9.7061     -3.2383
           shrink_at_coupled_pooling    -0.2247     -0.3287     -0.0748    -0.2835     -0.3643     -0.2122
pca16      coupled_vs_lw                -0.0562     -0.1231     -0.0055    -0.0434     -0.0568     -0.0307
           normalisation                 0.0028     -0.0051      0.0130    -0.0004     -0.0016      0.0005
           pooling_at_lw_shrink         -0.1165     -0.1736     -0.0632    -0.0817     -0.0924     -0.0688
           pooling_at_coupled_shrink     0.1052     -0.0148      0.1472     0.0188     -0.0076      0.0394
           shrink_at_no_pooling         -0.1643     -0.2113     -0.1032    -0.0618     -0.0786     -0.0464
           shrink_at_coupled_pooling     0.0575      0.0382      0.0804     0.0387      0.0296      0.0517
native     coupled_vs_lw              -396.5990   -455.5755   -368.4657  -554.5579   -635.1728   -509.4551
           normalisation                 3.2013      2.7706      3.6229     2.2187      2.0943      2.3233
           pooling_at_lw_shrink        262.0106    251.4484    266.1253   254.5140    249.1082    257.7100
           pooling_at_coupled_shrink  5713.3653   2154.5380  14401.1116  7039.7865   1856.2566  19834.2795
           shrink_at_no_pooling      -5561.2310 -14279.1334  -1955.9535 -6961.5922 -19791.3173  -1730.7598
           shrink_at_coupled_pooling  -658.3947   -722.5344   -630.8229  -802.0316   -887.2221   -760.6882
native_alt coupled_vs_lw              -563.6251   -612.6978   -527.3368  -945.6443  -1009.3588   -876.1451
           normalisation                 3.2056      2.7748      3.6281     2.2221      2.0977      2.3269
           pooling_at_lw_shrink        262.3020    251.7188    266.4056   254.8486    249.4503    258.0742
           pooling_at_coupled_shrink  3908.3976   2130.9796   5079.0038  4124.6648   1851.7361   5899.3649
           shrink_at_no_pooling      -3751.0582  -4969.2454  -1931.9743 -4038.8487  -5871.6664  -1726.0202
           shrink_at_coupled_pooling  -829.1327   -882.4960   -794.1090 -1202.7150  -1260.9068  -1135.0932

Folds with 95 % interval above / below 0:

                                      macro_above  macro_below  micro_above  micro_below
space      contrast                                                                     
pca64      coupled_vs_lw                        3            0            0            2
           normalisation                        5            0            5            0
           pooling_at_lw_shrink                 4            0            4            0
           pooling_at_coupled_shrink            5            0            5            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            0            3            0            5
pca16      coupled_vs_lw                        0            3            0            5
           normalisation                        0            1            0            1
           pooling_at_lw_shrink                 0            5            0            5
           pooling_at_coupled_shrink            3            0            1            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            5            0            5            0
native     coupled_vs_lw                        0            5            0            5
           normalisation                        5            0            5            0
           pooling_at_lw_shrink                 5            0            5            0
           pooling_at_coupled_shrink            5            0            5            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            0            5            0            5
native_alt coupled_vs_lw                        0            5            0            5
           normalisation                        5            0            5            0
           pooling_at_lw_shrink                 5            0            5            0
           pooling_at_coupled_shrink            5            0            5            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            0            5            0            5

By class group:

     space                  contrast  group  class_folds  taxa  class_diff_mean  class_diff_median  maha_part_mean  logdet_part_mean  class_folds_better  fold_macro_mean  fold_macro_min  fold_macro_max  folds_ci_above_0  folds_ci_below_0
     pca64             coupled_vs_lw  15-63           25     7           2.9660             3.0118          7.5565           -4.5905                  20           2.9194          0.7421          4.7426                 4                 0
     pca64             coupled_vs_lw 64-255          119    26          -0.0410            -0.0572          0.8075           -0.8485                  55          -0.0412         -0.0919         -0.0110                 0                 0
     pca64             coupled_vs_lw  >=256           31     7          -0.1959            -0.1823         -0.1736           -0.0223                   3          -0.1926         -0.2956         -0.0022                 0                 4
     pca64             coupled_vs_lw   n<=p           25     7           2.9660             3.0118          7.5565           -4.5905                  20           2.9194          0.7421          4.7426                 4                 0
     pca64             coupled_vs_lw    n>p          150    31          -0.0730            -0.0943          0.6048           -0.6778                  58          -0.0733         -0.1082         -0.0327                 0                 0
     pca64             normalisation  15-63           25     7           0.2970             0.1885          1.3413           -1.0443                  21           0.2938          0.1385          0.4318                 5                 0
     pca64             normalisation 64-255          119    26           0.0141             0.0121          0.2403           -0.2262                  89           0.0143          0.0085          0.0218                 5                 0
     pca64             normalisation  >=256           31     7           0.0054             0.0052          0.1061           -0.1007                  26           0.0054          0.0036          0.0060                 5                 0
     pca64             normalisation   n<=p           25     7           0.2970             0.1885          1.3413           -1.0443                  21           0.2938          0.1385          0.4318                 5                 0
     pca64             normalisation    n>p          150    31           0.0123             0.0089          0.2125           -0.2002                 115           0.0124          0.0075          0.0180                 5                 0
     pca64      pooling_at_lw_shrink  15-63           25     7           2.7300             1.6548         13.2764          -10.5465                  19           2.7262          0.8846          4.3679                 4                 0
     pca64      pooling_at_lw_shrink 64-255          119    26           0.1907             0.1570          4.6621           -4.4714                  68           0.1957          0.0165          0.4150                 3                 0
     pca64      pooling_at_lw_shrink  >=256           31     7           0.0682             0.2096          3.0206           -2.9524                  19           0.0698         -0.0065          0.2088                 1                 0
     pca64      pooling_at_lw_shrink   n<=p           25     7           2.7300             1.6548         13.2764          -10.5465                  19           2.7262          0.8846          4.3679                 4                 0
     pca64      pooling_at_lw_shrink    n>p          150    31           0.1654             0.1670          4.3229           -4.1575                  87           0.1674          0.0120          0.3203                 3                 0
     pca64 pooling_at_coupled_shrink  15-63           22     7          74.0857            13.6182         94.2294          -20.1436                  19          78.2061         15.2660        277.9899                 5                 0
     pca64 pooling_at_coupled_shrink 64-255          119    26           4.5226             2.8379         11.2889           -6.7664                 102           4.5394          3.6721          5.1834                 5                 0
     pca64 pooling_at_coupled_shrink  >=256           31     7           1.3374             0.8956          5.2132           -3.8758                  27           1.3368          1.1507          1.4340                 5                 0
     pca64 pooling_at_coupled_shrink   n<=p           22     7          74.0857            13.6182         94.2294          -20.1436                  19          78.2061         15.2660        277.9899                 5                 0
     pca64 pooling_at_coupled_shrink    n>p          150    31           3.8643             2.4270         10.0333           -6.1690                 129           3.8705          3.1841          4.2593                 5                 0
     pca64      shrink_at_no_pooling  15-63           22     7         -71.4091           -10.7217        -87.5300           16.1209                   3         -75.4713       -274.3540        -10.8937                 0                 5
     pca64      shrink_at_no_pooling 64-255          119    26          -4.5776            -2.8256        -10.7216            6.1440                   6          -4.5949         -5.2223         -3.7264                 0                 5
     pca64      shrink_at_no_pooling  >=256           31     7          -1.5387            -1.0879         -5.4929            3.9542                   3          -1.5348         -1.6564         -1.3829                 0                 5
     pca64      shrink_at_no_pooling   n<=p           22     7         -71.4091           -10.7217        -87.5300           16.1209                   3         -75.4713       -274.3540        -10.8937                 0                 5
     pca64      shrink_at_no_pooling    n>p          150    31          -3.9496            -2.3124         -9.6410            5.6914                   9          -3.9562         -4.3616         -3.2729                 0                 5
     pca64 shrink_at_coupled_pooling  15-63           25     7          -0.0610             0.6038         -7.0613            7.0002                  16          -0.1006         -0.7320          0.2581                 0                 0
     pca64 shrink_at_coupled_pooling 64-255          119    26          -0.2458            -0.0698         -4.0948            3.8491                  47          -0.2512         -0.4539         -0.0708                 0                 3
     pca64 shrink_at_coupled_pooling  >=256           31     7          -0.2696            -0.1921         -3.3003            3.0307                  12          -0.2678         -0.3240         -0.2170                 0                 5
     pca64 shrink_at_coupled_pooling   n<=p           25     7          -0.0610             0.6038         -7.0613            7.0002                  16          -0.1006         -0.7320          0.2581                 0                 0
     pca64 shrink_at_coupled_pooling    n>p          150    31          -0.2507            -0.0843         -3.9306            3.6799                  59          -0.2531         -0.4226         -0.1008                 0                 4
     pca16             coupled_vs_lw  15-63           25     7          -0.0776            -0.1707          1.3308           -1.4084                   8          -0.1203         -0.5101          0.2122                 0                 1
     pca16             coupled_vs_lw 64-255          119    26          -0.0559            -0.0415          0.2878           -0.3437                  32          -0.0557         -0.0825         -0.0402                 0                 4
     pca16             coupled_vs_lw  >=256           31     7          -0.0381            -0.0336          0.1021           -0.1403                   9          -0.0383         -0.0557         -0.0267                 0                 4
     pca16             coupled_vs_lw   n<=p            4     2           0.5879             0.1903          3.2853           -2.6974                   2           0.4553         -0.3297          0.9854                 0                 0
     pca16             coupled_vs_lw    n>p          171    35          -0.0709            -0.0462          0.3365           -0.4074                  47          -0.0706         -0.1231         -0.0402                 0                 4
     pca16             normalisation  15-63           25     7           0.0283             0.0002          0.2894           -0.2611                  13           0.0230         -0.0303          0.0902                 1                 1
     pca16             normalisation 64-255          119    26          -0.0015            -0.0008          0.0550           -0.0565                  51          -0.0015         -0.0028         -0.0005                 0                 1
     pca16             normalisation  >=256           31     7          -0.0004            -0.0006          0.0248           -0.0252                  11          -0.0004         -0.0009          0.0002                 0                 1
     pca16             normalisation   n<=p            4     2           0.1934             0.1077          0.7365           -0.5430                   3           0.1649          0.0392          0.2792                 1                 0
     pca16             normalisation    n>p          171    35          -0.0015            -0.0007          0.0679           -0.0694                  72          -0.0015         -0.0051          0.0012                 0                 1
     pca16      pooling_at_lw_shrink  15-63           25     7          -0.2751            -0.4271          1.6214           -1.8965                   5          -0.3127         -0.6619         -0.0146                 0                 2
     pca16      pooling_at_lw_shrink 64-255          119    26          -0.1000            -0.0719          0.5581           -0.6581                  27          -0.0996         -0.1243         -0.0784                 0                 5
     pca16      pooling_at_lw_shrink  >=256           31     7          -0.0508            -0.0591          0.3329           -0.3838                   7          -0.0513         -0.0655         -0.0362                 0                 5
     pca16      pooling_at_lw_shrink   n<=p            4     2           0.2486            -0.0002          2.6197           -2.3712                   2           0.1657         -0.6706          0.6702                 0                 0
     pca16      pooling_at_lw_shrink    n>p          171    35          -0.1248            -0.0805          0.6245           -0.7493                  37          -0.1245         -0.1736         -0.0847                 0                 5
     pca16 pooling_at_coupled_shrink  15-63           25     7           0.7971             0.2137          3.4649           -2.6678                  14           0.7874          0.1596          1.2381                 4                 0
     pca16 pooling_at_coupled_shrink 64-255          119    26          -0.0079            -0.0188          0.7886           -0.7964                  49          -0.0074         -0.0417          0.0143                 0                 0
     pca16 pooling_at_coupled_shrink  >=256           31     7          -0.0186            -0.0190          0.4118           -0.4304                  11          -0.0193         -0.0343          0.0011                 0                 0
     pca16 pooling_at_coupled_shrink   n<=p            4     2           0.2424            -0.0220          2.9006           -2.6582                   2           0.1543         -0.2576          0.5068                 0                 0
     pca16 pooling_at_coupled_shrink    n>p          171    35           0.1020            -0.0143          1.0621           -0.9601                  72           0.1027         -0.0148          0.1591                 3                 0
     pca16      shrink_at_no_pooling  15-63           25     7          -0.9030            -0.1113         -2.4234            1.5204                   7          -0.9307         -1.4951         -0.6394                 0                 5
     pca16      shrink_at_no_pooling 64-255          119    26          -0.0465            -0.0261         -0.5558            0.5093                  37          -0.0468         -0.0601         -0.0288                 0                 4
     pca16      shrink_at_no_pooling  >=256           31     7          -0.0191            -0.0026         -0.3345            0.3153                  13          -0.0186         -0.0349         -0.0088                 0                 1
     pca16      shrink_at_no_pooling   n<=p            4     2           0.1520             0.1406         -0.3518            0.5038                   2           0.1362         -0.1113          0.3204                 1                 0
     pca16      shrink_at_no_pooling    n>p          171    35          -0.1714            -0.0247         -0.7935            0.6221                  55          -0.1718         -0.2143         -0.1032                 0                 5
     pca16 shrink_at_coupled_pooling  15-63           25     7           0.1693             0.2122         -0.5799            0.7492                  18           0.1694          0.0917          0.2959                 2                 0
     pca16 shrink_at_coupled_pooling 64-255          119    26           0.0456             0.0262         -0.3253            0.3709                  74           0.0455          0.0350          0.0700                 5                 0
     pca16 shrink_at_coupled_pooling  >=256           31     7           0.0131             0.0101         -0.2556            0.2687                  23           0.0134          0.0025          0.0203                 2                 0
     pca16 shrink_at_coupled_pooling   n<=p            4     2           0.1459             0.0865         -0.0709            0.2167                   2           0.1248         -0.1361          0.3017                 1                 1
     pca16 shrink_at_coupled_pooling    n>p          171    35           0.0554             0.0262         -0.3559            0.4113                 113           0.0554          0.0339          0.0728                 4                 0
    native             coupled_vs_lw  15-63           25     7         356.9303           445.9313        107.9733          248.9570                  22         363.3784        294.8063        496.6181                 5                 0
    native             coupled_vs_lw 64-255          119    26        -518.8774           123.5608       -865.3224          346.4450                  76        -521.7721       -631.9048       -445.5790                 0                 5
    native             coupled_vs_lw  >=256           26     6        -561.2450            -9.1987       -935.4845          374.2395                  13        -560.7434       -629.8180       -413.9509                 0                 5
    native             coupled_vs_lw   n<=p          170    35        -396.5619           115.9125       -732.9214          336.3595                 111        -396.5990       -455.5755       -368.4657                 0                 5
    native             coupled_vs_lw    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
    native             normalisation  15-63           25     7           9.1791             8.0359         25.8717          -16.6926                  24           9.2213          6.8338         11.9962                 5                 0
    native             normalisation 64-255          119    26           2.3979             1.9633          6.0132           -3.6153                 118           2.4049          2.2017          2.6018                 5                 0
    native             normalisation  >=256           31     7           1.4851             1.4129          3.0941           -1.6090                  31           1.4844          1.4145          1.5427                 5                 0
    native             normalisation   n<=p          175    36           3.2050             1.9193          8.3331           -5.1281                 173           3.2013          2.7706          3.6229                 5                 0
    native             normalisation    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
    native      pooling_at_lw_shrink  15-63           25     7         313.3444           284.9697        474.6831         -161.3387                  25         316.9935        288.9035        376.8822                 5                 0
    native      pooling_at_lw_shrink 64-255          119    26         255.9717           248.1087        373.6079         -117.6362                 119         256.3839        242.3265        267.1322                 5                 0
    native      pooling_at_lw_shrink  >=256           31     7         243.7860           241.2783        345.0313         -101.2453                  31         243.7977        238.4548        246.0378                 5                 0
    native      pooling_at_lw_shrink   n<=p          175    36         262.0092           248.9256        382.9851         -120.9759                 175         262.0106        251.4484        266.1253                 5                 0
    native      pooling_at_lw_shrink    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
    native pooling_at_coupled_shrink  15-63           22     7        2393.7300           975.4525       2752.5255         -358.7955                  22        2432.4668       1123.2316       4621.5749                 5                 0
    native pooling_at_coupled_shrink 64-255           89    21        3655.3697          1107.9511       3958.0725         -302.7028                  89        3661.0507       1692.6835       5212.5060                 5                 0
    native pooling_at_coupled_shrink  >=256           20     5       18957.4317          2176.6477      19377.2240         -419.7923                  20       18688.2453       1790.2343      69821.9604                 5                 0
    native pooling_at_coupled_shrink   n<=p          131    29        5779.6840          1204.2159       6109.6832         -329.9992                 131        5713.3653       2154.5380      14401.1116                 5                 0
    native pooling_at_coupled_shrink    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
    native      shrink_at_no_pooling  15-63           22     7       -1981.4079          -525.3341      -2533.6218          552.2139                   0       -2015.0031      -4136.9530       -725.9285                 0                 5
    native      shrink_at_no_pooling 64-255           89    21       -3521.4327         -1033.9128      -4031.1428          509.7100                   5       -3527.9981      -5083.4757      -1533.6855                 0                 5
    native      shrink_at_no_pooling  >=256           20     5      -19012.0343         -2165.1367     -19692.5431          680.5089                   0      -18734.9041     -69974.0900      -1722.3185                 0                 5
    native      shrink_at_no_pooling   n<=p          131    29       -5627.7799         -1048.5697      -6170.7042          542.9243                   5       -5561.2310     -14279.1334      -1955.9535                 0                 5
    native      shrink_at_no_pooling    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
    native shrink_at_coupled_pooling  15-63           25     7          34.4068            88.2269       -392.5815          426.9883                  20          37.1636        -29.9370        107.7396                 2                 0
    native shrink_at_coupled_pooling 64-255          119    26        -777.2470           -94.2735      -1244.9435          467.6965                  45        -780.5610       -901.6388       -690.1073                 0                 5
    native shrink_at_coupled_pooling  >=256           26     6        -780.3079          -266.6236      -1262.6650          482.3571                   0        -779.5139       -853.8504       -631.1638                 0                 5
    native shrink_at_coupled_pooling   n<=p          170    35        -658.3543           -91.2010      -1122.3065          463.9522                  65        -658.3947       -722.5344       -630.8229                 0                 5
    native shrink_at_coupled_pooling    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
native_alt             coupled_vs_lw  15-63           25     7         357.3013           445.3001        109.6058          247.6954                  22         363.8268        294.7263        498.6098                 5                 0
native_alt             coupled_vs_lw 64-255          119    26        -519.8682           124.9795       -866.2905          346.4223                  76        -522.8029       -634.9565       -446.4982                 0                 5
native_alt             coupled_vs_lw  >=256           31     7       -1474.1486          -159.1260      -1892.4801          418.3315                  13       -1478.8574      -1688.8307      -1332.8853                 0                 5
native_alt             coupled_vs_lw   n<=p          175    36        -563.6022           100.7705       -908.6589          345.0567                 111        -563.6251       -612.6978       -527.3368                 0                 5
native_alt             coupled_vs_lw    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
native_alt             normalisation  15-63           25     7           9.1896             8.0632         25.8822          -16.6926                  24           9.2318          6.8466         12.0054                 5                 0
native_alt             normalisation 64-255          119    26           2.4012             1.9641          6.0165           -3.6153                 118           2.4082          2.2052          2.6048                 5                 0
native_alt             normalisation  >=256           31     7           1.4886             1.4175          3.0976           -1.6090                  31           1.4879          1.4179          1.5462                 5                 0
native_alt             normalisation   n<=p          175    36           3.2093             1.9158          8.3374           -5.1281                 173           3.2056          2.7748          3.6281                 5                 0
native_alt             normalisation    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
native_alt      pooling_at_lw_shrink  15-63           25     7         313.5578           284.8651        474.8896         -161.3318                  25         317.1983        289.1514        377.0709                 5                 0
native_alt      pooling_at_lw_shrink 64-255          119    26         256.2276           248.1115        373.8923         -117.6647                 119         256.6402        242.5807        267.3769                 5                 0
native_alt      pooling_at_lw_shrink  >=256           31     7         244.2762           241.3706        345.5606         -101.2844                  31         244.2902        238.9353        246.6005                 5                 0
native_alt      pooling_at_lw_shrink   n<=p          175    36         262.3005           249.6356        383.3017         -121.0012                 175         262.3020        251.7188        266.4056                 5                 0
native_alt      pooling_at_lw_shrink    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
native_alt pooling_at_coupled_shrink  15-63           22     7        2347.6782           966.9414       2704.6502         -356.9720                  22        2383.7541       1110.2532       4451.2886                 5                 0
native_alt pooling_at_coupled_shrink 64-255           89    21        3873.3081          1100.1339       4178.0695         -304.7614                  89        3889.9043       1694.8891       5483.0600                 5                 0
native_alt pooling_at_coupled_shrink  >=256           19     5        5879.9128          1554.1382       6237.0972         -357.1844                  19        5380.0191       1485.2336      13772.3393                 5                 0
native_alt pooling_at_coupled_shrink   n<=p          130    29        3908.3976          1155.6024       4229.6564         -321.2589                 130        3908.3976       2130.9796       5079.0038                 5                 0
native_alt pooling_at_coupled_shrink    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
native_alt      shrink_at_no_pooling  15-63           22     7       -1934.8331          -518.0208      -2483.7919          548.9589                   0       -1965.7361      -3964.6842       -712.7699                 0                 5
native_alt      shrink_at_no_pooling 64-255           89    21       -3740.6676         -1023.4542      -4252.5168          511.8492                   5       -3758.2241      -5393.3083      -1535.6941                 0                 5
native_alt      shrink_at_no_pooling  >=256           19     5       -5902.7277         -1480.4140      -6502.4637          599.7360                   0       -5392.1399     -13899.6111      -1454.9252                 0                 5
native_alt      shrink_at_no_pooling   n<=p          130    29       -3751.0582         -1038.7087      -4282.0325          530.9743                   5       -3751.0582      -4969.2454      -1931.9743                 0                 5
native_alt      shrink_at_no_pooling    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
native_alt shrink_at_coupled_pooling  15-63           25     7          34.5538            88.3484       -391.1659          425.7197                  20          37.3966        -30.0953        109.5335                 2                 0
native_alt shrink_at_coupled_pooling 64-255          119    26        -778.4970           -93.2264      -1246.1993          467.7023                  45        -781.8513       -904.9381       -691.2841                 0                 5
native_alt shrink_at_coupled_pooling  >=256           31     7       -1719.9134          -398.8095      -2241.1383          521.2249                   0       -1724.6354      -1936.8879      -1578.2533                 0                 5
native_alt shrink_at_coupled_pooling   n<=p          175    36        -829.1121          -100.2446      -1300.2980          471.1860                  65        -829.1327       -882.4960       -794.1090                 0                 5
native_alt shrink_at_coupled_pooling    n>p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0

Parameters:

     space stratum  class_folds  n_le_p_share  coupled_alpha_keep_median  coupled_shrink_to_identity_median  lw_alpha_shrink_median  coupled_beta_median  share_beta_0  share_beta_1  share_alpha_1  share_gamma_clipped  share_kappa_floored  residual_alpha_max  residual_beta_max  lw_condition_median  coupled_condition_median
     pca64  64-255          119          0.00                     0.9792                             0.0208                  0.0685               0.9092           0.0        0.0336         0.1933                  0.0               0.0756                 0.0                0.0             175.3007                  246.1081
     pca64   >=256           31          0.00                     0.9950                             0.0050                  0.0392               0.9289           0.0        0.0000         0.4516                  0.0               0.0000                 0.0                0.0             259.0145                  301.1637
     pca64   15-63           25          1.00                     0.8988                             0.1012                  0.2333               0.7249           0.0        0.0000         0.1200                  0.0               0.6800                 0.0                0.0              60.4622                   82.5892
     pca16  64-255          119          0.00                     0.9830                             0.0170                  0.0507               0.9343           0.0        0.0252         0.2269                  0.0               0.1092                 0.0                0.0              49.1607                   50.3958
     pca16   >=256           31          0.00                     0.9956                             0.0044                  0.0262               0.9473           0.0        0.0000         0.1935                  0.0               0.1290                 0.0                0.0              56.3057                   56.4415
     pca16   15-63           25          0.16                     0.9241                             0.0759                  0.1723               0.7801           0.0        0.0000         0.1600                  0.0               0.4000                 0.0                0.0              22.4374                   24.0349
    native  64-255          119          1.00                     0.9758                             0.0242                  0.0981               0.8716           0.0        0.0000         0.2521                  0.0               0.3025                 0.0                0.0            1905.5856                 7176.7136
    native   >=256           31          1.00                     0.9945                             0.0055                  0.0538               0.9036           0.0        0.0000         0.3548                  0.0               0.2581                 0.0                0.0            3893.4803                27452.2734
    native   15-63           25          1.00                     0.8886                             0.1114                  0.2961               0.6390           0.0        0.0000         0.1200                  0.0               0.8000                 0.0                0.0             502.0027                 1358.6867
native_alt  64-255          119          1.00                     0.9760                             0.0240                  0.0978               0.8715           0.0        0.0000         0.2521                  0.0               0.3025                 0.0                0.0            1905.2746                 7199.2442
native_alt   >=256           31          1.00                     0.9947                             0.0053                  0.0537               0.9036           0.0        0.0000         0.3871                  0.0               0.2581                 0.0                0.0            3903.2837                28224.4803
native_alt   15-63           25          1.00                     0.8876                             0.1124                  0.2961               0.6390           0.0        0.0000         0.1200                  0.0               0.8000                 0.0                0.0             507.4199                 1346.9263

Corners (diagnostic; minus LW):

     space stratum  class_folds  class_scm_defined  coupled_minus_lw_mean  pooled_scm_minus_lw_mean  class_scm_minus_lw_mean  class_identity_minus_lw_mean  coupled_beats_lw  pooled_scm_beats_lw  class_scm_beats_lw  class_identity_beats_lw
     pca64  64-255          119                119                -0.0410                  -16.2170                -259.9811                      -29.4362                55                    0                   0                        0
     pca64   >=256           31                 31                -0.1959                  -14.0226                  -1.9620                      -32.5965                 3                    0                   0                        0
     pca64   15-63           25                  0                 2.9660                   -4.7664                      NaN                      -19.7131                20                    5                   0                        0
     pca16  64-255          119                119                -0.0559                   -4.0860                  -0.1639                       -5.8174                32                    0                  25                        0
     pca16   >=256           31                 31                -0.0381                   -3.5460                  -0.0270                       -6.3142                 9                    0                  13                        0
     pca16   15-63           25                 21                -0.0776                   -3.1219                  -8.2816                       -5.2431                 8                    1                   1                        0
    native  64-255          119                  0              -518.8774                  289.6017                      NaN                     -586.0533                76                  114                   0                        0
    native   >=256           31                  0              -561.2450                  307.5409                      NaN                     -706.1345                13                   27                   0                        0
    native   15-63           25                  0               356.9303                  545.3008                      NaN                     -321.4793                22                   25                   0                        0
native_alt  64-255          119                  0              -519.8682                  289.7869                      NaN                     -585.9980                76                  114                   0                        0
native_alt   >=256           31                  0             -1474.1486                  308.1602                      NaN                     -705.8403                13                   27                   0                        0
native_alt   15-63           25                  0               357.3013                  545.4702                      NaN                     -321.4185                22                   25                   0                        0

LW reference vs lw_fit_diagnostics:

{
 "spaces": [
  "pca16",
  "pca64"
 ],
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

Examples (coupled_vs_lw):

     space            why         taxon_group       diff  n_train   beta  folds  lw_shrink  coupled_shrink   diff_min   diff_max
     pca64   largest_gain           Sialis_sp    10.5694     15.5 0.6572    2.0     0.4664          0.4284     5.5501    15.5887
     pca64   largest_loss  Lepidostoma_hirtum    -1.3605    170.0 0.9352    5.0     0.0495          0.0038    -2.0646    -0.8727
     pca64 smallest_class       Leuctra_nigra     3.8481     15.0 0.4568    3.0     0.4875          0.2581     3.0118     4.8384
     pca64  largest_class          Simuliidae    -0.2136    564.0 0.9793    5.0     0.0216          0.0156    -0.4920     0.4346
     pca64   most_pooling       Leuctra_nigra     3.8481     15.0 0.4568    3.0     0.4875          0.2581     3.0118     4.8384
     pca16   largest_gain           Sialis_sp     1.6410     15.5 0.8205    2.0     0.4383          0.4883     0.7103     2.5718
     pca16   largest_loss       Leuctra_nigra    -0.7863     15.0 0.6487    3.0     0.3506          0.2706    -1.4284    -0.3297
     pca16 smallest_class       Leuctra_nigra    -0.7863     15.0 0.6487    3.0     0.3506          0.2706    -1.4284    -0.3297
     pca16  largest_class          Simuliidae    -0.0491    564.0 0.9830    5.0     0.0153          0.0137    -0.0821     0.0153
     pca16   most_pooling       Leuctra_nigra    -0.7863     15.0 0.6487    3.0     0.3506          0.2706    -1.4284    -0.3297
    native   largest_gain    Plectrocnemia_sp   500.4712     39.0 0.7185    5.0     0.1902          0.0339   446.9903   582.3601
    native   largest_loss Micrasema_setiferum -7986.5189    206.0 0.9571    5.0     0.0283          0.0000 -8621.4288 -7185.8072
    native smallest_class       Leuctra_nigra   438.1318     15.0 0.3420    3.0     0.5478          0.2591   391.0910   497.4454
    native  largest_class          Simuliidae    71.3351    564.0 0.9673    5.0     0.0302          0.0207    60.8040    92.6398
    native   most_pooling       Leuctra_nigra   438.1318     15.0 0.3420    3.0     0.5478          0.2591   391.0910   497.4454
native_alt   largest_gain    Plectrocnemia_sp   502.2206     39.0 0.7188    5.0     0.1901          0.0343   447.8647   583.1193
native_alt   largest_loss Micrasema_setiferum -7990.6979    206.0 0.9571    5.0     0.0283          0.0000 -8625.4348 -7190.3073
native_alt smallest_class       Leuctra_nigra   438.4259     15.0 0.3415    3.0     0.5474          0.2587   391.3292   497.4642
native_alt  largest_class          Simuliidae    71.4138    564.0 0.9674    5.0     0.0301          0.0207    60.8033    92.6584
native_alt   most_pooling       Leuctra_nigra   438.4259     15.0 0.3415    3.0     0.5474          0.2587   391.3292   497.4642

Resources per fold:

{
 "fold0": {
  "total_seconds": 55.293986317992676,
  "max_rss_gb": 2.6163558959960938,
  "pca64_seconds": 0.7081982279778458,
  "pca16_seconds": 0.4893852110253647,
  "native_seconds": 22.32247751398245,
  "native_alt_seconds": 22.37894593202509
 },
 "fold1": {
  "total_seconds": 54.770593688997906,
  "max_rss_gb": 2.5969085693359375,
  "pca64_seconds": 0.6689999139634892,
  "pca16_seconds": 0.48504633200354874,
  "native_seconds": 21.84522218600614,
  "native_alt_seconds": 21.72382503701374
 },
 "fold2": {
  "total_seconds": 61.41553760599345,
  "max_rss_gb": 2.6135597229003906,
  "pca64_seconds": 0.7531267490121536,
  "pca16_seconds": 0.5171140630263835,
  "native_seconds": 25.122958631021902,
  "native_alt_seconds": 24.23653860297054
 },
 "fold3": {
  "total_seconds": 59.10356024495559,
  "max_rss_gb": 2.6001358032226562,
  "pca64_seconds": 0.7060989279998466,
  "pca16_seconds": 0.5349071900127456,
  "native_seconds": 24.416091888037045,
  "native_alt_seconds": 23.474894186016172
 },
 "fold4": {
  "total_seconds": 64.97879825899145,
  "max_rss_gb": 2.587757110595703,
  "pca64_seconds": 0.6225826370064169,
  "pca16_seconds": 0.47999433701625094,
  "native_seconds": 29.455182351986878,
  "native_alt_seconds": 24.31593464303296
 }
}

## AquaMonitor

Across folds (fold mean, min, max; within-space contrasts):

                                     macro_mean                       micro_mean                      
                                           mean        min        max       mean        min        max
space      contrast                                                                                   
pca64      coupled_vs_lw                 0.0503    -0.1561     0.3465    -0.0378    -0.0647     0.0064
           normalisation                 0.0291     0.0167     0.0496     0.0032     0.0015     0.0055
           pooling_at_lw_shrink          0.0292    -0.2406     0.3228    -0.0421    -0.0831     0.0155
           pooling_at_coupled_shrink    11.4690     7.2926    15.5942     1.1431     0.8979     1.3709
           shrink_at_no_pooling        -11.4798   -15.6407    -7.2426    -1.1876    -1.4406    -0.9289
           shrink_at_coupled_pooling    -0.0080    -0.0659     0.0678     0.0010    -0.0146     0.0194
pca16      coupled_vs_lw                -0.1480    -0.1735    -0.1115    -0.0247    -0.0279    -0.0180
           normalisation                -0.0011    -0.0045     0.0061    -0.0001    -0.0006     0.0006
           pooling_at_lw_shrink         -0.2035    -0.2257    -0.1663    -0.0355    -0.0409    -0.0277
           pooling_at_coupled_shrink    -0.0384    -0.1144     0.0306    -0.0146    -0.0191    -0.0101
           shrink_at_no_pooling         -0.1085    -0.1565    -0.0470    -0.0100    -0.0146    -0.0072
           shrink_at_coupled_pooling     0.0567     0.0425     0.0715     0.0109     0.0091     0.0136
native     coupled_vs_lw               -86.4971  -102.4324   -66.7277   -96.8671  -101.5108   -88.2938
           normalisation                 2.4599     2.0726     2.9435     0.3861     0.3491     0.4457
           pooling_at_lw_shrink        185.4033   180.0760   198.0809    42.9049    41.4491    45.3660
           pooling_at_coupled_shrink  3681.7676   879.7629  8848.1462  1119.7943   179.9514  3961.7282
           shrink_at_no_pooling      -3594.5206 -8780.2015  -767.4988 -1149.1347 -3998.7412  -199.5495
           shrink_at_coupled_pooling  -274.3602  -286.9291  -262.6716  -140.1582  -144.9563  -134.1055
native_alt coupled_vs_lw               -86.3988  -102.1528   -66.3244   -96.9085  -101.6465   -88.3059
           normalisation                 2.4582     2.0704     2.9412     0.3860     0.3490     0.4455
           pooling_at_lw_shrink        185.3644   180.0389   198.0338    42.9038    41.4414    45.3725
           pooling_at_coupled_shrink  3366.5562   883.1418  7141.1053   907.8152   180.2628  2887.5841
           shrink_at_no_pooling      -3279.0659 -7072.7629  -770.9074  -937.0962 -2924.4332  -199.8885
           shrink_at_coupled_pooling  -274.2214  -286.8232  -262.7252  -140.1982  -145.0822  -134.1239

Folds with 95 % interval above / below 0:

                                      macro_above  macro_below  micro_above  micro_below
space      contrast                                                                     
pca64      coupled_vs_lw                        1            0            0            3
           normalisation                        5            0            5            0
           pooling_at_lw_shrink                 0            0            0            2
           pooling_at_coupled_shrink            5            0            5            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            0            0            0            0
pca16      coupled_vs_lw                        0            5            0            5
           normalisation                        0            2            0            1
           pooling_at_lw_shrink                 0            5            0            5
           pooling_at_coupled_shrink            0            1            0            3
           shrink_at_no_pooling                 0            3            0            4
           shrink_at_coupled_pooling            5            0            5            0
native     coupled_vs_lw                        0            5            0            5
           normalisation                        5            0            5            0
           pooling_at_lw_shrink                 5            0            5            0
           pooling_at_coupled_shrink            5            0            5            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            0            5            0            5
native_alt coupled_vs_lw                        0            5            0            5
           normalisation                        5            0            5            0
           pooling_at_lw_shrink                 5            0            5            0
           pooling_at_coupled_shrink            5            0            5            0
           shrink_at_no_pooling                 0            5            0            5
           shrink_at_coupled_pooling            0            5            0            5

By class group:

     space                  contrast  group  class_folds  taxa  class_diff_mean  class_diff_median  maha_part_mean  logdet_part_mean  class_folds_better  fold_macro_mean  fold_macro_min  fold_macro_max  folds_ci_above_0  folds_ci_below_0
     pca64             coupled_vs_lw  15-63           39     9           1.0955             0.5477          4.0025           -2.9070                  28           1.0761          0.3191          1.6955                 4                 0
     pca64             coupled_vs_lw 64-255           95    19          -0.2660            -0.1856          1.2678           -1.5338                  36          -0.2660         -0.6316          0.1350                 0                 2
     pca64             coupled_vs_lw  >=256           70    14          -0.1018            -0.0220          0.4384           -0.5402                  15          -0.1018         -0.1372         -0.0812                 0                 2
     pca64             coupled_vs_lw   n<=p           39     9           1.0955             0.5477          4.0025           -2.9070                  28           1.0761          0.3191          1.6955                 4                 0
     pca64             coupled_vs_lw    n>p          165    33          -0.1964            -0.0615          0.9159           -1.1123                  51          -0.1964         -0.4021          0.0195                 0                 2
     pca64             normalisation  15-63           39     9           0.1049             0.0641          0.8419           -0.7371                  29           0.1034          0.0449          0.1824                 3                 0
     pca64             normalisation 64-255           95    19           0.0186             0.0141          0.3019           -0.2833                  67           0.0186          0.0039          0.0290                 3                 0
     pca64             normalisation  >=256           70    14           0.0012             0.0001          0.0436           -0.0424                  42           0.0012          0.0005          0.0019                 1                 0
     pca64             normalisation   n<=p           39     9           0.1049             0.0641          0.8419           -0.7371                  29           0.1034          0.0449          0.1824                 3                 0
     pca64             normalisation    n>p          165    33           0.0112             0.0011          0.1923           -0.1811                 109           0.0112          0.0026          0.0174                 4                 0
     pca64      pooling_at_lw_shrink  15-63           39     9           0.7514             0.4734          9.6581           -8.9067                  22           0.7409          0.2716          1.2882                 0                 0
     pca64      pooling_at_lw_shrink 64-255           95    19          -0.1330            -0.1650          5.6427           -5.7756                  40          -0.1330         -0.5186          0.2680                 0                 1
     pca64      pooling_at_lw_shrink  >=256           70    14          -0.1526            -0.0574          1.6054           -1.7580                  17          -0.1526         -0.1804         -0.1025                 0                 4
     pca64      pooling_at_lw_shrink   n<=p           39     9           0.7514             0.4734          9.6581           -8.9067                  22           0.7409          0.2716          1.2882                 0                 0
     pca64      pooling_at_lw_shrink    n>p          165    33          -0.1413            -0.0715          3.9299           -4.0712                  57          -0.1413         -0.3648          0.0887                 0                 1
     pca64 pooling_at_coupled_shrink  15-63           34     8          42.5182             9.6517         60.1359          -17.6177                  29          43.3821         21.1655         72.7551                 5                 0
     pca64 pooling_at_coupled_shrink 64-255           95    19           8.6823             2.5569         17.6658           -8.9836                  75           8.6823          7.4245         10.1951                 5                 0
     pca64 pooling_at_coupled_shrink  >=256           70    14           0.1109             0.0071          2.1514           -2.0405                  38           0.1109          0.0502          0.1772                 1                 0
     pca64 pooling_at_coupled_shrink   n<=p           34     8          42.5182             9.6517         60.1359          -17.6177                  29          43.3821         21.1655         72.7551                 5                 0
     pca64 pooling_at_coupled_shrink    n>p          165    33           5.0459             0.3082         11.0840           -6.0380                 113           5.0459          4.3499          5.9146                 5                 0
     pca64      shrink_at_no_pooling  15-63           34     8         -41.5735            -5.7195        -56.7732           15.1996                   7         -42.4574        -72.5099        -20.0006                 0                 5
     pca64      shrink_at_no_pooling 64-255           95    19          -8.9668            -2.1373        -16.7000            7.7331                   9          -8.9668        -10.0891         -7.6871                 0                 5
     pca64      shrink_at_no_pooling  >=256           70    14          -0.2139            -0.0291         -1.7566            1.5427                  13          -0.2139         -0.2603         -0.1444                 0                 5
     pca64      shrink_at_no_pooling   n<=p           34     8         -41.5735            -5.7195        -56.7732           15.1996                   7         -42.4574        -72.5099        -20.0006                 0                 5
     pca64      shrink_at_no_pooling    n>p          165    33          -5.2535            -0.4457        -10.3604            5.1069                  22          -5.2535         -5.9125         -4.5363                 0                 5
     pca64 shrink_at_coupled_pooling  15-63           39     9           0.2393             0.1166         -6.4974            6.7367                  27           0.2318         -0.0592          0.5121                 0                 0
     pca64 shrink_at_coupled_pooling 64-255           95    19          -0.1516             0.0720         -4.6768            4.5252                  57          -0.1516         -0.3301         -0.0453                 0                 0
     pca64 shrink_at_coupled_pooling  >=256           70    14           0.0496             0.0056         -1.2106            1.2601                  46           0.0496          0.0156          0.0862                 2                 0
     pca64 shrink_at_coupled_pooling   n<=p           39     9           0.2393             0.1166         -6.4974            6.7367                  27           0.2318         -0.0592          0.5121                 0                 0
     pca64 shrink_at_coupled_pooling    n>p          165    33          -0.0662             0.0181         -3.2063            3.1400                 103          -0.0662         -0.1535         -0.0178                 0                 0
     pca16             coupled_vs_lw  15-63           39     9          -0.2616            -0.2245          0.8481           -1.1097                  10          -0.2648         -0.3913         -0.1641                 0                 3
     pca16             coupled_vs_lw 64-255           95    19          -0.1888            -0.1635          0.4985           -0.6872                  13          -0.1888         -0.2551         -0.1479                 0                 5
     pca16             coupled_vs_lw  >=256           70    14          -0.0291            -0.0042          0.1551           -0.1842                  20          -0.0291         -0.0356         -0.0228                 0                 5
     pca16             coupled_vs_lw   n<=p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
     pca16             coupled_vs_lw    n>p          204    42          -0.1479            -0.0807          0.4475           -0.5954                  43          -0.1480         -0.1735         -0.1115                 0                 5
     pca16             normalisation  15-63           39     9          -0.0014            -0.0134          0.1829           -0.1843                  10          -0.0019         -0.0222          0.0266                 0                 1
     pca16             normalisation 64-255           95    19          -0.0017            -0.0004          0.0691           -0.0708                  44          -0.0017         -0.0061          0.0020                 0                 1
     pca16             normalisation  >=256           70    14          -0.0001            -0.0000          0.0105           -0.0106                  31          -0.0001         -0.0003          0.0001                 0                 0
     pca16             normalisation   n<=p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
     pca16             normalisation    n>p          204    42          -0.0011            -0.0002          0.0708           -0.0718                  85          -0.0011         -0.0045          0.0061                 0                 2
     pca16      pooling_at_lw_shrink  15-63           39     9          -0.3865            -0.3502          1.0732           -1.4596                   6          -0.3890         -0.4888         -0.2432                 0                 4
     pca16      pooling_at_lw_shrink 64-255           95    19          -0.2446            -0.2178          0.6985           -0.9430                  12          -0.2446         -0.3248         -0.1924                 0                 5
     pca16      pooling_at_lw_shrink  >=256           70    14          -0.0457            -0.0076          0.2392           -0.2850                  18          -0.0457         -0.0576         -0.0336                 0                 5
     pca16      pooling_at_lw_shrink   n<=p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
     pca16      pooling_at_lw_shrink    n>p          204    42          -0.2035            -0.1360          0.6125           -0.8160                  36          -0.2035         -0.2257         -0.1663                 0                 5
     pca16 pooling_at_coupled_shrink  15-63           39     9           0.2225            -0.0984          2.3422           -2.1197                  16           0.2245         -0.1681          0.5902                 1                 0
     pca16 pooling_at_coupled_shrink 64-255           95    19          -0.1505            -0.1475          0.9787           -1.1292                  23          -0.1505         -0.2366         -0.0924                 0                 3
     pca16 pooling_at_coupled_shrink  >=256           70    14          -0.0317            -0.0033          0.2767           -0.3085                  23          -0.0317         -0.0490         -0.0181                 0                 3
     pca16 pooling_at_coupled_shrink   n<=p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
     pca16 pooling_at_coupled_shrink    n>p          204    42          -0.0384            -0.0559          0.9985           -1.0369                  62          -0.0384         -0.1144          0.0306                 0                 1
     pca16      shrink_at_no_pooling  15-63           39     9          -0.4827            -0.1817         -1.6770            1.1943                   9          -0.4875         -0.7741         -0.1249                 0                 3
     pca16      shrink_at_no_pooling 64-255           95    19          -0.0366            -0.0041         -0.5494            0.5128                  46          -0.0366         -0.0584         -0.0125                 0                 0
     pca16      shrink_at_no_pooling  >=256           70    14           0.0027             0.0002         -0.1321            0.1349                  39           0.0027         -0.0055          0.0180                 1                 0
     pca16      shrink_at_no_pooling   n<=p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
     pca16      shrink_at_no_pooling    n>p          204    42          -0.1084            -0.0004         -0.6218            0.5134                  94          -0.1085         -0.1565         -0.0470                 0                 3
     pca16 shrink_at_coupled_pooling  15-63           39     9           0.1262             0.0582         -0.4080            0.5342                  23           0.1261          0.0213          0.2457                 4                 0
     pca16 shrink_at_coupled_pooling 64-255           95    19           0.0575             0.0446         -0.2691            0.3266                  72           0.0575          0.0424          0.0757                 5                 0
     pca16 shrink_at_coupled_pooling  >=256           70    14           0.0168             0.0033         -0.0946            0.1114                  53           0.0168          0.0107          0.0266                 5                 0
     pca16 shrink_at_coupled_pooling   n<=p            0     0              NaN                NaN             NaN               NaN                   0              NaN             NaN             NaN                 0                 0
     pca16 shrink_at_coupled_pooling    n>p          204    42           0.0567             0.0283         -0.2358            0.2925                 148           0.0567          0.0425          0.0715                 5                 0
    native             coupled_vs_lw  15-63           39     9         292.7033           280.5347        -52.9853          345.6886                  37         290.9733        223.5059        385.5841                 5                 0
    native             coupled_vs_lw 64-255           95    19        -135.9443           113.3959       -476.5430          340.5987                  64        -135.9443       -157.9049       -117.4488                 0                 5
    native             coupled_vs_lw  >=256           70    14        -230.4312           -34.0322       -386.8862          156.4550                   8        -230.4312       -248.9264       -211.6796                 0                 5
    native             coupled_vs_lw   n<=p          169    35         -94.4067           113.3959       -415.3771          320.9704                 109         -94.5222       -114.0449        -70.8873                 0                 5
    native             coupled_vs_lw    n>p           35     7         -47.8496           -21.4734       -120.6092           72.7596                   0         -47.8496        -52.0881        -42.7501                 0                 5
    native             normalisation  15-63           39     9           5.9595             4.7617         17.7409          -11.7813                  37           5.9031          3.7047          7.7368                 5                 0
    native             normalisation 64-255           95    19           2.5528             2.4151          7.0816           -4.5288                  94           2.5528          2.2735          2.7964                 5                 0
    native             normalisation  >=256           70    14           0.3896             0.2188          1.0668           -0.6771                  70           0.3896          0.3677          0.4039                 5                 0
    native             normalisation   n<=p          169    35           2.9625             2.1088          8.4662           -5.5037                 166           2.9598          2.5029          3.5399                 5                 0
    native             normalisation    n>p           35     7           0.0443             0.0504          0.2438           -0.1996                  35           0.0443          0.0366          0.0473                 5                 0
    native      pooling_at_lw_shrink  15-63           39     9         274.0819           252.7223        396.7194         -122.6376                  39         273.0166        231.4706        304.7037                 5                 0
    native      pooling_at_lw_shrink 64-255           95    19         234.0294           228.2435        339.1760         -105.1466                  95         234.0294        221.3288        246.9812                 5                 0
    native      pooling_at_lw_shrink  >=256           70    14          70.0724            43.8436        115.1762          -45.1038                  70          70.0724         66.1504         72.7536                 5                 0
    native      pooling_at_lw_shrink   n<=p          169    35         221.9114           211.5278        325.2002         -103.2888                 169         221.8822        215.5407        236.9255                 5                 0
    native      pooling_at_lw_shrink    n>p           35     7           9.2580             7.6147         22.7792          -13.5213                  35           9.2580          7.8192          9.9131                 5                 0
    native pooling_at_coupled_shrink  15-63           33     8        6305.9322           575.7730       6671.4638         -365.5316                  33        6341.4500       2030.7323      11824.2553                 5                 0
    native pooling_at_coupled_shrink 64-255           71    15        2546.2337           676.0007       2759.1031         -212.8693                  71        2511.2733        848.0784       4993.4642                 5                 0
    native pooling_at_coupled_shrink  >=256           63    13        3607.1024            72.1733       3702.7353          -95.6329                  63        3507.2260        341.2434      15273.5805                 5                 0
    native pooling_at_coupled_shrink   n<=p          132    29        4659.4029           658.7400       4905.9539         -246.5511                 132        4648.2458       1118.5254      11221.7806                 5                 0
    native pooling_at_coupled_shrink    n>p           35     7          30.9893            13.5470         49.7436          -18.7543                  35          30.9893         27.0395         33.3187                 5                 0
    native      shrink_at_no_pooling  15-63           33     8       -6012.0606          -199.0210      -6657.7633          645.7026                   0       -6050.2415     -11620.6393      -1739.8120                 0                 5
    native      shrink_at_no_pooling 64-255           71    15       -2407.9589          -438.7487      -2822.3329          414.3740                   5       -2372.4459      -4893.8685       -673.3875                 0                 5
    native      shrink_at_no_pooling  >=256           63    13       -3686.6809          -160.4382      -3903.7638          217.0829                   0       -3586.1330     -15381.1500       -391.1388                 0                 5
    native      shrink_at_no_pooling   n<=p          132    29       -4536.8415          -432.1850      -5000.4398          463.5984                   5       -4524.9931     -11122.6922       -962.8471                 0                 5
    native      shrink_at_no_pooling    n>p           35     7         -78.8832           -34.9959       -170.5967           91.7135                   0         -78.8832        -85.4205        -69.8262                 0                 5
    native shrink_at_coupled_pooling  15-63           39     9          12.6619            32.8173       -467.4456          480.1075                  24          12.0536        -11.6694         73.1436                 1                 0
    native shrink_at_coupled_pooling 64-255           95    19        -372.5265           -30.4645       -822.8006          450.2741                  38        -372.5265       -400.1458       -341.1268                 0                 5
    native shrink_at_coupled_pooling  >=256           70    14        -300.8932           -94.1260       -503.1292          202.2360                   0        -300.8932       -322.0838       -282.8723                 0                 5
    native shrink_at_coupled_pooling   n<=p          169    35        -319.2806           -64.6794       -749.0435          429.7629                  62        -319.3642       -333.4915       -304.3656                 0                 5
    native shrink_at_coupled_pooling    n>p           35     7         -57.1518           -29.1447       -143.6323           86.4805                   0         -57.1518        -61.9444        -50.6058                 0                 5
native_alt             coupled_vs_lw  15-63           39     9         292.3860           281.2586        -53.3102          345.6962                  37         290.6575        223.2442        385.2040                 5                 0
native_alt             coupled_vs_lw 64-255           95    19        -135.4737           113.0893       -475.9117          340.4380                  64        -135.4737       -157.3194       -117.3891                 0                 5
native_alt             coupled_vs_lw  >=256           70    14        -230.6093           -33.9366       -387.1616          156.5523                   8        -230.6093       -249.3674       -211.9701                 0                 5
native_alt             coupled_vs_lw   n<=p          169    35         -94.2577           113.0893       -415.1534          320.8957                 109         -94.3718       -113.6678        -70.3703                 0                 5
native_alt             coupled_vs_lw    n>p           35     7         -48.0015           -21.5196       -120.8885           72.8870                   0         -48.0015        -52.2708        -42.8571                 0                 5
native_alt             normalisation  15-63           39     9           5.9530             4.7536         17.7344          -11.7813                  37           5.8966          3.6938          7.7256                 5                 0
native_alt             normalisation 64-255           95    19           2.5518             2.4124          7.0806           -4.5288                  94           2.5518          2.2722          2.7962                 5                 0
native_alt             normalisation  >=256           70    14           0.3896             0.2184          1.0667           -0.6771                  70           0.3896          0.3676          0.4039                 5                 0
native_alt             normalisation   n<=p          169    35           2.9604             2.1091          8.4641           -5.5037                 166           2.9577          2.5003          3.5371                 5                 0
native_alt             normalisation    n>p           35     7           0.0442             0.0504          0.2438           -0.1996                  35           0.0442          0.0366          0.0473                 5                 0
native_alt      pooling_at_lw_shrink  15-63           39     9         273.9349           252.5345        396.5374         -122.6025                  39         272.8655        231.1577        304.3837                 5                 0
native_alt      pooling_at_lw_shrink 64-255           95    19         233.9991           228.4402        339.1547         -105.1557                  95         233.9991        221.2478        246.9973                 5                 0
native_alt      pooling_at_lw_shrink  >=256           70    14          70.0822            43.7983        115.1886          -45.1064                  70          70.0822         66.1459         72.7294                 5                 0
native_alt      pooling_at_lw_shrink   n<=p          169    35         221.8643           211.4717        325.1498         -103.2856                 169         221.8351        215.4947        236.8679                 5                 0
native_alt      pooling_at_lw_shrink    n>p           35     7           9.2589             7.6245         22.7868          -13.5279                  35           9.2589          7.8253          9.9111                 5                 0
native_alt pooling_at_coupled_shrink  15-63           33     8        6477.0812           575.9337       6843.4601         -366.3789                  33        6510.2207       2052.6896      12061.3561                 5                 0
native_alt pooling_at_coupled_shrink 64-255           71    15        2547.8450           673.0108       2760.4948         -212.6498                  71        2510.5008        845.2901       5161.9376                 5                 0
native_alt pooling_at_coupled_shrink  >=256           63    13        2691.6377            72.7199       2786.0763          -94.4387                  63        2619.9893        342.5283      10847.7526                 5                 0
native_alt pooling_at_coupled_shrink   n<=p          132    29        4266.1064           657.1793       4512.1758         -246.0694                 132        4247.9619       1122.8309       9055.1200                 5                 0
native_alt pooling_at_coupled_shrink    n>p           35     7          31.0801            13.5592         49.8547          -18.7746                  35          31.0801         27.1093         33.4103                 5                 0
native_alt      shrink_at_no_pooling  15-63           33     8       -6183.5843          -199.5956      -6830.1226          646.5383                   0       -6219.3809     -11858.0333      -1762.0168                 0                 5
native_alt      shrink_at_no_pooling 64-255           71    15       -2408.9633          -440.6484      -2822.9644          414.0011                   5       -2371.0597      -5062.2113       -670.4534                 0                 5
native_alt      shrink_at_no_pooling  >=256           63    13       -2771.0560          -160.5583      -2987.0566          216.0006                   0       -2698.7445     -10954.5925       -392.5490                 0                 5
native_alt      shrink_at_no_pooling   n<=p          132    29       -4143.1955          -430.6817      -4606.2465          463.0510                   5       -4124.3615      -8955.4785       -967.1606                 0                 5
native_alt      shrink_at_no_pooling    n>p           35     7         -79.1258           -35.0648       -170.9870           91.8612                   0         -79.1258        -85.6966        -70.0030                 0                 5
native_alt shrink_at_coupled_pooling  15-63           39     9          12.4981            32.9765       -467.5820          480.0801                  24          11.8954        -11.6073         73.0947                 1                 0
native_alt shrink_at_coupled_pooling 64-255           95    19        -372.0246           -30.3244       -822.1470          450.1225                  38        -372.0246       -398.9171       -340.9842                 0                 5
native_alt shrink_at_coupled_pooling  >=256           70    14        -301.0811           -93.6320       -503.4170          202.3359                   0        -301.0811       -322.5004       -283.1860                 0                 5
native_alt shrink_at_coupled_pooling   n<=p          169    35        -319.0824           -64.7046       -748.7674          429.6850                  62        -319.1646       -333.0850       -304.4015                 0                 5
native_alt shrink_at_coupled_pooling    n>p           35     7         -57.3046           -29.2008       -143.9192           86.6145                   0         -57.3046        -62.1230        -50.7189                 0                 5

Parameters:

     space stratum  class_folds  n_le_p_share  coupled_alpha_keep_median  coupled_shrink_to_identity_median  lw_alpha_shrink_median  coupled_beta_median  share_beta_0  share_beta_1  share_alpha_1  share_gamma_clipped  share_kappa_floored  residual_alpha_max  residual_beta_max  lw_condition_median  coupled_condition_median
     pca64   >=256           70           0.0                     0.9980                             0.0020                  0.0081               0.9906           0.0           0.0         0.1286                  0.0               0.0000                 0.0                0.0             417.9627                  405.8665
     pca64   15-63           39           1.0                     0.9418                             0.0582                  0.1536               0.8151           0.0           0.0         0.1282                  0.0               0.3077                 0.0                0.0             126.2894                  145.4083
     pca64  64-255           95           0.0                     0.9649                             0.0351                  0.0727               0.8987           0.0           0.0         0.2632                  0.0               0.1158                 0.0                0.0             224.9354                  264.8852
     pca16   >=256           70           0.0                     0.9978                             0.0022                  0.0062               0.9921           0.0           0.0         0.2143                  0.0               0.0000                 0.0                0.0              90.5640                   74.0414
     pca16   15-63           39           0.0                     0.9192                             0.0808                  0.1211               0.8519           0.0           0.0         0.2308                  0.0               0.2308                 0.0                0.0              48.7265                   29.9425
     pca16  64-255           95           0.0                     0.9640                             0.0360                  0.0567               0.9339           0.0           0.0         0.2105                  0.0               0.0947                 0.0                0.0              52.9571                   41.7987
    native   >=256           70           0.5                     0.9973                             0.0027                  0.0112               0.9860           0.0           0.0         0.1000                  0.0               0.0000                 0.0                0.0           21686.8714                83368.5808
    native   15-63           39           1.0                     0.9348                             0.0652                  0.2125               0.7826           0.0           0.0         0.1538                  0.0               0.6923                 0.0                0.0             922.1917                 3105.7630
    native  64-255           95           1.0                     0.9665                             0.0335                  0.1005               0.8637           0.0           0.0         0.2526                  0.0               0.4421                 0.0                0.0            2379.8479                 6649.4535
native_alt   >=256           70           0.5                     0.9973                             0.0027                  0.0111               0.9860           0.0           0.0         0.1000                  0.0               0.0000                 0.0                0.0           21696.6534                83009.8583
native_alt   15-63           39           1.0                     0.9344                             0.0656                  0.2125               0.7825           0.0           0.0         0.1538                  0.0               0.6923                 0.0                0.0             922.4392                 3122.5113
native_alt  64-255           95           1.0                     0.9664                             0.0336                  0.1007               0.8638           0.0           0.0         0.2526                  0.0               0.4526                 0.0                0.0            2378.4626                 6645.5723

Corners (diagnostic; minus LW):

     space stratum  class_folds  class_scm_defined  coupled_minus_lw_mean  pooled_scm_minus_lw_mean  class_scm_minus_lw_mean  class_identity_minus_lw_mean  coupled_beats_lw  pooled_scm_beats_lw  class_scm_beats_lw  class_identity_beats_lw
     pca64   >=256           70                 70                -0.1018                  -22.2758                  -0.7037                      -40.3595                15                    0                   7                        0
     pca64   15-63           39                  0                 1.0955                  -15.8030                      NaN                      -30.0567                28                    0                   0                        0
     pca64  64-255           95                 95                -0.2660                  -20.2609                 -60.0371                      -33.5209                36                    0                   0                        0
     pca16   >=256           70                 70                -0.0291                   -4.7985                  -0.0058                       -6.9042                20                    0                  34                        0
     pca16   15-63           39                 39                -0.2616                   -5.0160                  -3.3039                       -6.4706                10                    1                   4                        0
     pca16  64-255           95                 95                -0.1888                   -4.6293                  -0.2482                       -6.3405                13                    0                  27                        0
    native   >=256           70                 35              -230.4312                  -69.6398                -211.0764                    -1187.0855                 8                   22                   0                        0
    native   15-63           39                  0               292.7033                  287.4921                      NaN                     -521.6082                37                   32                   0                        0
    native  64-255           95                  0              -135.9443                  176.2281                      NaN                     -683.7887                64                   71                   0                        0
native_alt   >=256           70                 35              -230.6093                  -69.7248                -211.1834                    -1187.2468                 8                   22                   0                        0
native_alt   15-63           39                  0               292.3860                  287.3110                      NaN                     -521.8313                37                   32                   0                        0
native_alt  64-255           95                  0              -135.4737                  176.1199                      NaN                     -683.8807                64                   71                   0                        0

LW reference vs lw_fit_diagnostics:

{
 "spaces": [
  "pca16",
  "pca64"
 ],
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

Examples (coupled_vs_lw):

     space            why            taxon_group       diff  n_train   beta  folds  lw_shrink  coupled_shrink   diff_min   diff_max
     pca64   largest_gain             Mystacides     3.1210     37.5 0.7071    4.0     0.2241          0.0379     1.3314     4.7117
     pca64   largest_loss Heptagenia dalecarlica    -3.1039    137.0 0.8987    5.0     0.0615          0.0000    -4.6348    -2.3376
     pca64 smallest_class                 Sisyra     2.1951     35.0 0.7320    3.0     0.3535          0.2782    -0.2366     6.6050
     pca64  largest_class           Chironomidae     0.0013   7416.0 0.9980    5.0     0.0009          0.0002    -0.0046     0.0090
     pca64   most_pooling                Pedicia     2.1073     40.0 0.6856    5.0     0.1445          0.0256     0.0075     4.7026
     pca16   largest_gain        Cloeon dipterum     0.0172    103.0 0.9339    5.0     0.1070          0.0767    -0.0888     0.1230
     pca16   largest_loss                Pedicia    -0.6602     40.0 0.7620    5.0     0.1046          0.0095    -1.1889     0.0013
     pca16 smallest_class                 Sisyra    -0.0169     35.0 0.7828    3.0     0.3093          0.3187    -0.7626     1.2490
     pca16  largest_class           Chironomidae     0.0000   7416.0 0.9985    5.0     0.0006          0.0001    -0.0013     0.0013
     pca16   most_pooling             Mystacides    -0.1542     37.5 0.7299    4.0     0.2168          0.0730    -0.4435     0.0480
    native   largest_gain       Caenis rivulorum   534.5899     36.0 0.8718    3.0     0.0860          0.0128   409.9709   672.8812
    native   largest_loss Heptagenia dalecarlica -3113.2276    137.0 0.8669    5.0     0.0809          0.0000 -3433.4692 -2495.0144
    native smallest_class                 Sisyra   289.1093     35.0 0.6564    3.0     0.3996          0.2430   238.8269   347.9664
    native  largest_class           Chironomidae    -3.3611   7416.0 0.9975    5.0     0.0011          0.0002    -3.9974    -2.7634
    native   most_pooling                Pedicia   461.8601     40.0 0.6389    5.0     0.1962          0.0337   398.5905   514.3582
native_alt   largest_gain       Caenis rivulorum   533.3926     36.0 0.8717    3.0     0.0860          0.0127   408.5448   671.8433
native_alt   largest_loss Heptagenia dalecarlica -3113.1636    137.0 0.8669    5.0     0.0808          0.0000 -3434.3934 -2494.3959
native_alt smallest_class                 Sisyra   289.5264     35.0 0.6564    3.0     0.3993          0.2428   238.9498   348.3706
native_alt  largest_class           Chironomidae    -3.3532   7416.0 0.9975    5.0     0.0011          0.0002    -3.9725    -2.7544
native_alt   most_pooling                Pedicia   462.0456     40.0 0.6388    5.0     0.1967          0.0343   397.8050   513.0314

Resources per fold:

{
 "fold0": {
  "total_seconds": 193.33795921399724,
  "max_rss_gb": 10.303462982177734,
  "pca64_seconds": 0.9851947569986805,
  "pca16_seconds": 0.7086453679949045,
  "native_seconds": 29.86541520798346,
  "native_alt_seconds": 31.101535398047417
 },
 "fold1": {
  "total_seconds": 230.63422718201764,
  "max_rss_gb": 10.303180694580078,
  "pca64_seconds": 1.128916805027984,
  "pca16_seconds": 0.8333693940076046,
  "native_seconds": 46.13725163700292,
  "native_alt_seconds": 33.04513892199611
 },
 "fold2": {
  "total_seconds": 266.58816630998626,
  "max_rss_gb": 10.248764038085938,
  "pca64_seconds": 0.9983420459902845,
  "pca16_seconds": 0.6953081879764795,
  "native_seconds": 38.94371197902365,
  "native_alt_seconds": 33.67794550204417
 },
 "fold3": {
  "total_seconds": 244.44205765996594,
  "max_rss_gb": 10.277397155761719,
  "pca64_seconds": 1.2218100530444644,
  "pca16_seconds": 0.8171117780148052,
  "native_seconds": 40.56445566401817,
  "native_alt_seconds": 37.50511684600497
 },
 "fold4": {
  "total_seconds": 260.2276494259713,
  "max_rss_gb": 10.242382049560547,
  "pca64_seconds": 1.1338597189751454,
  "pca16_seconds": 0.8056278320145793,
  "native_seconds": 44.597844985022675,
  "native_alt_seconds": 45.48613869497785
 }
}

