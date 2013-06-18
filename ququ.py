import sys
import os
import math
from sklearn import metrics
from sklearn import tree
from sklearn import cross_validation
import StringIO, pydot
import pylab as pl
import random
import matplotlib.pyplot as plt
from numpy import median
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import scoreatpercentile
sys.path.append("/afs/inf.ed.ac.uk/user/s09/s0954584/Desktop/pypoker")
sys.path.append("/afs/inf.ed.ac.uk/user/s09/s0954584/Desktop/pypoker/.libs")
from pokereval import PokerEval
import pickle
from sklearn.cross_validation import StratifiedKFold
from scipy import interp

pe = PokerEval()
smallBlind = 50
with open("tripHandVals", 'r') as f:
  trips = pickle.load(f)

convLog = """
processed/azure_sky
7164
102.342857143 70 Mean AUC : 0.907
455.425714286 15 Mean AUC : 0.914
808.508571429 8 Mean AUC : 0.917
1161.59142857 6 Mean AUC : 0.922
1514.67428571 4 Mean AUC : 0.921
1867.75714286 3 Mean AUC : 0.923
2220.84 3 Mean AUC : 0.927
2573.92285714 2 Mean AUC : 0.940
2927.00571429 2 Mean AUC : 0.929
3280.08857143 2 Mean AUC : 0.941
3633.17142857 1 Mean AUC : 0.940
3986.25428571 1 Mean AUC : 0.941
4339.33714286 1 Mean AUC : 0.946
4692.42 1 Mean AUC : 0.949
5045.50285714 1 Mean AUC : 0.948
5398.58571429 1 Mean AUC : 0.952
5751.66857143 1 Mean AUC : 0.954
6104.75142857 1 Mean AUC : 0.956
6457.83428571 1 Mean AUC : 0.959
6810.91714286 1 Mean AUC : 0.961
7164.0 1 Mean AUC : 0.965
processed/dcubot
8896
127.085714286 70 Mean AUC : 0.547
565.531428571 15 Mean AUC : 0.570
1003.97714286 8 Mean AUC : 0.593
1442.42285714 6 Mean AUC : 0.604
1880.86857143 4 Mean AUC : 0.619
2319.31428571 3 Mean AUC : 0.631
2757.76 3 Mean AUC : 0.645
3196.20571429 2 Mean AUC : 0.656
3634.65142857 2 Mean AUC : 0.705
4073.09714286 2 Mean AUC : 0.711
4511.54285714 1 Mean AUC : 0.687
4949.98857143 1 Mean AUC : 0.722
5388.43428571 1 Mean AUC : 0.716
5826.88 1 Mean AUC : 0.754
6265.32571429 1 Mean AUC : 0.767
6703.77142857 1 Mean AUC : 0.783
7142.21714286 1 Mean AUC : 0.796
7580.66285714 1 Mean AUC : 0.810
8019.10857143 1 Mean AUC : 0.829
8457.55428571 1 Mean AUC : 0.830
8896.0 1 Mean AUC : 0.853
8896.0 1 Mean AUC : 0.852
processed/hugh
6696
95.6571428571 70 Mean AUC : 0.658
425.674285714 15 Mean AUC : 0.685
755.691428571 8 Mean AUC : 0.703
1085.70857143 6 Mean AUC : 0.723
1415.72571429 4 Mean AUC : 0.758
1745.74285714 3 Mean AUC : 0.778
2075.76 3 Mean AUC : 0.786
2405.77714286 2 Mean AUC : 0.793
2735.79428571 2 Mean AUC : 0.813
3065.81142857 2 Mean AUC : 0.818
3395.82857143 1 Mean AUC : 0.837
3725.84571429 1 Mean AUC : 0.850
4055.86285714 1 Mean AUC : 0.861
4385.88 1 Mean AUC : 0.850
4715.89714286 1 Mean AUC : 0.871
5045.91428571 1 Mean AUC : 0.877
5375.93142857 1 Mean AUC : 0.888
5705.94857143 1 Mean AUC : 0.896
6035.96571429 1 Mean AUC : 0.899
6365.98285714 1 Mean AUC : 0.910
6696.0 1 Mean AUC : 0.924
processed/hyperborean
5940
84.8571428571 70 Mean AUC : 0.549
377.614285714 15 Mean AUC : 0.577
670.371428571 8 Mean AUC : 0.599
963.128571429 6 Mean AUC : 0.622
1255.88571429 4 Mean AUC : 0.630
1548.64285714 3 Mean AUC : 0.645
1841.4 3 Mean AUC : 0.667
2134.15714286 2 Mean AUC : 0.704
2426.91428571 2 Mean AUC : 0.693
2719.67142857 2 Mean AUC : 0.723
3012.42857143 1 Mean AUC : 0.724
3305.18571429 1 Mean AUC : 0.757
3597.94285714 1 Mean AUC : 0.767
3890.7 1 Mean AUC : 0.782
4183.45714286 1 Mean AUC : 0.781
4476.21428571 1 Mean AUC : 0.814
4768.97142857 1 Mean AUC : 0.808
5061.72857143 1 Mean AUC : 0.822
5354.48571429 1 Mean AUC : 0.839
5647.24285714 1 Mean AUC : 0.845
5940.0 1 Mean AUC : 0.867
5940.0 1 Mean AUC : 0.865
processed/little_rock
8476
121.085714286 70 Mean AUC : 0.541
538.831428571 15 Mean AUC : 0.583
956.577142857 8 Mean AUC : 0.595
1374.32285714 6 Mean AUC : 0.622
1792.06857143 4 Mean AUC : 0.641
2209.81428571 3 Mean AUC : 0.654
2627.56 3 Mean AUC : 0.668
3045.30571429 2 Mean AUC : 0.694
3463.05142857 2 Mean AUC : 0.688
3880.79714286 2 Mean AUC : 0.714
4298.54285714 1 Mean AUC : 0.726
4716.28857143 1 Mean AUC : 0.733
5134.03428571 1 Mean AUC : 0.759
5551.78 1 Mean AUC : 0.766
5969.52571429 1 Mean AUC : 0.780
6387.27142857 1 Mean AUC : 0.806
6805.01714286 1 Mean AUC : 0.820
7222.76285714 1 Mean AUC : 0.818
7640.50857143 1 Mean AUC : 0.838
8058.25428571 1 Mean AUC : 0.846
8476.0 1 Mean AUC : 0.853
processed/spewy_louie
4200
60.0 70 Mean AUC : 0.531
267.0 15 Mean AUC : 0.574
474.0 8 Mean AUC : 0.572
681.0 6 Mean AUC : 0.589
888.0 4 Mean AUC : 0.606
1095.0 3 Mean AUC : 0.618
1302.0 3 Mean AUC : 0.639
1509.0 2 Mean AUC : 0.638
1716.0 2 Mean AUC : 0.649
1923.0 2 Mean AUC : 0.679
2130.0 1 Mean AUC : 0.688
2337.0 1 Mean AUC : 0.693
2544.0 1 Mean AUC : 0.711
2751.0 1 Mean AUC : 0.721
2958.0 1 Mean AUC : 0.755
3165.0 1 Mean AUC : 0.747
3372.0 1 Mean AUC : 0.776
3579.0 1 Mean AUC : 0.790
3786.0 1 Mean AUC : 0.802
3993.0 1 Mean AUC : 0.818
4200.0 1 Mean AUC : 0.830
4200.0 1 Mean AUC : 0.826
processed/neo_poker_lab
8880
126.857142857 70 Mean AUC : 0.573
564.514285714 15 Mean AUC : 0.626
1002.17142857 8 Mean AUC : 0.658
1439.82857143 6 Mean AUC : 0.690
1877.48571429 4 Mean AUC : 0.726
2315.14285714 3 Mean AUC : 0.737
2752.8 3 Mean AUC : 0.755
3190.45714286 2 Mean AUC : 0.786
3628.11428571 2 Mean AUC : 0.795
4065.77142857 2 Mean AUC : 0.811
4503.42857143 1 Mean AUC : 0.833
4941.08571429 1 Mean AUC : 0.846
5378.74285714 1 Mean AUC : 0.837
5816.4 1 Mean AUC : 0.862
6254.05714286 1 Mean AUC : 0.884
6691.71428571 1 Mean AUC : 0.885
7129.37142857 1 Mean AUC : 0.898
7567.02857143 1 Mean AUC : 0.904
8004.68571429 1 Mean AUC : 0.915
8442.34285714 1 Mean AUC : 0.920
8880.0 1 Mean AUC : 0.928
8880.0 1 Mean AUC : 0.938
processed/tartanian5
6788
96.9714285714 70 Mean AUC : 0.548
431.522857143 15 Mean AUC : 0.604
766.074285714 8 Mean AUC : 0.638
1100.62571429 6 Mean AUC : 0.656
1435.17714286 4 Mean AUC : 0.689
1769.72857143 3 Mean AUC : 0.693
2104.28 3 Mean AUC : 0.728
2438.83142857 2 Mean AUC : 0.746
2773.38285714 2 Mean AUC : 0.762
3107.93428571 2 Mean AUC : 0.772
3442.48571429 1 Mean AUC : 0.778
3777.03714286 1 Mean AUC : 0.799
4111.58857143 1 Mean AUC : 0.809
4446.14 1 Mean AUC : 0.822
4780.69142857 1 Mean AUC : 0.836
5115.24285714 1 Mean AUC : 0.848
5449.79428571 1 Mean AUC : 0.861
5784.34571429 1 Mean AUC : 0.859
6118.89714286 1 Mean AUC : 0.881
6453.44857143 1 Mean AUC : 0.889
6788.0 1 Mean AUC : 0.903
processed/sartre
1684
24.0571428571 70 Mean AUC : 0.786
107.054285714 15 Mean AUC : 0.833
190.051428571 8 Mean AUC : 0.861
273.048571429 6 Mean AUC : 0.889
356.045714286 4 Mean AUC : 0.892
439.042857143 3 Mean AUC : 0.917
522.04 3 Mean AUC : 0.907
605.037142857 2 Mean AUC : 0.904
688.034285714 2 Mean AUC : 0.912
771.031428571 2 Mean AUC : 0.926
854.028571429 1 Mean AUC : 0.923
937.025714286 1 Mean AUC : 0.930
1020.02285714 1 Mean AUC : 0.930
1103.02 1 Mean AUC : 0.937
1186.01714286 1 Mean AUC : 0.957
1269.01428571 1 Mean AUC : 0.944
1352.01142857 1 Mean AUC : 0.949
1435.00857143 1 Mean AUC : 0.951
1518.00571429 1 Mean AUC : 0.959
1601.00285714 1 Mean AUC : 0.959
1684.0 1 Mean AUC : 0.976
1684.0 1 Mean AUC : 0.971
processed/lucky7_12
5940
84.8571428571 70 Mean AUC : 0.618
377.614285714 15 Mean AUC : 0.669
670.371428571 8 Mean AUC : 0.683
963.128571429 6 Mean AUC : 0.711
1255.88571429 4 Mean AUC : 0.718
1548.64285714 3 Mean AUC : 0.742
1841.4 3 Mean AUC : 0.755
2134.15714286 2 Mean AUC : 0.772
2426.91428571 2 Mean AUC : 0.776
2719.67142857 2 Mean AUC : 0.790
3012.42857143 1 Mean AUC : 0.795
3305.18571429 1 Mean AUC : 0.822
3597.94285714 1 Mean AUC : 0.802
3890.7 1 Mean AUC : 0.818
4183.45714286 1 Mean AUC : 0.836
4476.21428571 1 Mean AUC : 0.852
4768.97142857 1 Mean AUC : 0.849
5061.72857143 1 Mean AUC : 0.868
5354.48571429 1 Mean AUC : 0.878
5647.24285714 1 Mean AUC : 0.884
5940.0 1 Mean AUC : 0.892
5940.0 1 Mean AUC : 0.892
processed/uni_mb_poker
6540
93.4285714286 70 Mean AUC : 0.690
415.757142857 15 Mean AUC : 0.775
738.085714286 8 Mean AUC : 0.817
1060.41428571 6 Mean AUC : 0.836
1382.74285714 4 Mean AUC : 0.886
1705.07142857 3 Mean AUC : 0.897
2027.4 3 Mean AUC : 0.913
2349.72857143 2 Mean AUC : 0.927
2672.05714286 2 Mean AUC : 0.935
2994.38571429 2 Mean AUC : 0.944
3316.71428571 1 Mean AUC : 0.957
3639.04285714 1 Mean AUC : 0.970
3961.37142857 1 Mean AUC : 0.969
4283.7 1 Mean AUC : 0.974
4606.02857143 1 Mean AUC : 0.979
4928.35714286 1 Mean AUC : 0.976
5250.68571429 1 Mean AUC : 0.981
5573.01428571 1 Mean AUC : 0.980
5895.34285714 1 Mean AUC : 0.984
6217.67142857 1 Mean AUC : 0.987
6540.0 1 Mean AUC : 0.985
6540.0 1 Mean AUC : 0.990
processed/all
84324
1204.62857143 70 Mean AUC : 0.645
5360.59714286 15 Mean AUC : 0.686
9516.56571429 8 Mean AUC : 0.709
13672.5342857 6 Mean AUC : 0.728
17828.5028571 4 Mean AUC : 0.743
21984.4714286 3 Mean AUC : 0.758
26140.44 3 Mean AUC : 0.769
30296.4085714 2 Mean AUC : 0.782
34452.3771429 2 Mean AUC : 0.802
38608.3457143 2 Mean AUC : 0.806
42764.3142857 1 Mean AUC : 0.818
46920.2828571 1 Mean AUC : 0.821
51076.2514286 1 Mean AUC : 0.832
55232.22 1 Mean AUC : 0.845
59388.1885714 1 Mean AUC : 0.856
63544.1571429 1 Mean AUC : 0.864
67700.1257143 1 Mean AUC : 0.873
71856.0942857 1 Mean AUC : 0.878
76012.0628571 1 Mean AUC : 0.891
80168.0314286 1 Mean AUC : 0.898
84324.0 1 Mean AUC : 0.905
"""

convLog = dict([(x.split('\n')[0],(x.split('\n')[1],[y.split(' ')[0] for y in x.split('\n')[2:-1]],[y.split(' ')[-1] for y in x.split('\n')[2:-1]])) for x in convLog.split("processed/")])

smallConvLog = """
processed/azure_sky
7164
10 716 Mean AUC : 0.866
84 85 Mean AUC : 0.907
158 45 Mean AUC : 0.905
232 30 Mean AUC : 0.910
306 23 Mean AUC : 0.907
380 18 Mean AUC : 0.909
454 15 Mean AUC : 0.918
528 13 Mean AUC : 0.912
602 11 Mean AUC : 0.910
676 10 Mean AUC : 0.911
750 9 Mean AUC : 0.913
824 8 Mean AUC : 0.914
898 7 Mean AUC : 0.913
972 7 Mean AUC : 0.917
1046 6 Mean AUC : 0.915
1120 6 Mean AUC : 0.914
1194 6 Mean AUC : 0.918
1268 5 Mean AUC : 0.914
1342 5 Mean AUC : 0.916
1416 5 Mean AUC : 0.915
1490 4 Mean AUC : 0.915
1500 4 Mean AUC : 0.915
processed/dcubot
8896
10 889 Mean AUC : 0.528
84 105 Mean AUC : 0.537
158 56 Mean AUC : 0.538
232 38 Mean AUC : 0.547
306 29 Mean AUC : 0.559
380 23 Mean AUC : 0.553
454 19 Mean AUC : 0.548
528 16 Mean AUC : 0.565
602 14 Mean AUC : 0.569
676 13 Mean AUC : 0.570
750 11 Mean AUC : 0.559
824 10 Mean AUC : 0.575
898 9 Mean AUC : 0.588
972 9 Mean AUC : 0.581
1046 8 Mean AUC : 0.583
1120 7 Mean AUC : 0.578
1194 7 Mean AUC : 0.588
1268 7 Mean AUC : 0.587
1342 6 Mean AUC : 0.586
1416 6 Mean AUC : 0.595
1490 5 Mean AUC : 0.583
1500 5 Mean AUC : 0.590
processed/hugh
6696
10 669 Mean AUC : 0.602
84 79 Mean AUC : 0.647
158 42 Mean AUC : 0.649
232 28 Mean AUC : 0.656
306 21 Mean AUC : 0.669
380 17 Mean AUC : 0.673
454 14 Mean AUC : 0.682
528 12 Mean AUC : 0.676
602 11 Mean AUC : 0.686
676 9 Mean AUC : 0.679
750 8 Mean AUC : 0.692
824 8 Mean AUC : 0.683
898 7 Mean AUC : 0.699
972 6 Mean AUC : 0.692
1046 6 Mean AUC : 0.707
1120 5 Mean AUC : 0.719
1194 5 Mean AUC : 0.701
1268 5 Mean AUC : 0.723
1342 4 Mean AUC : 0.717
1416 4 Mean AUC : 0.728
1490 4 Mean AUC : 0.719
1500 4 Mean AUC : 0.720
processed/hyperborean
5940
10 594 Mean AUC : 0.514
84 70 Mean AUC : 0.554
158 37 Mean AUC : 0.549
232 25 Mean AUC : 0.566
306 19 Mean AUC : 0.562
380 15 Mean AUC : 0.566
454 13 Mean AUC : 0.580
528 11 Mean AUC : 0.564
602 9 Mean AUC : 0.597
676 8 Mean AUC : 0.579
750 7 Mean AUC : 0.588
824 7 Mean AUC : 0.599
898 6 Mean AUC : 0.593
972 6 Mean AUC : 0.606
1046 5 Mean AUC : 0.603
1120 5 Mean AUC : 0.599
1194 4 Mean AUC : 0.614
1268 4 Mean AUC : 0.625
1342 4 Mean AUC : 0.620
1416 4 Mean AUC : 0.617
1490 3 Mean AUC : 0.641
1500 3 Mean AUC : 0.630
processed/little_rock
8476
10 847 Mean AUC : 0.518
84 100 Mean AUC : 0.543
158 53 Mean AUC : 0.555
232 36 Mean AUC : 0.549
306 27 Mean AUC : 0.553
380 22 Mean AUC : 0.551
454 18 Mean AUC : 0.556
528 16 Mean AUC : 0.568
602 14 Mean AUC : 0.578
676 12 Mean AUC : 0.579
750 11 Mean AUC : 0.573
824 10 Mean AUC : 0.582
898 9 Mean AUC : 0.595
972 8 Mean AUC : 0.584
1046 8 Mean AUC : 0.592
1120 7 Mean AUC : 0.594
1194 7 Mean AUC : 0.595
1268 6 Mean AUC : 0.604
1342 6 Mean AUC : 0.611
1416 5 Mean AUC : 0.611
1490 5 Mean AUC : 0.612
1500 5 Mean AUC : 0.616
processed/spewy_louie
4200
10 420 Mean AUC : 0.517
84 50 Mean AUC : 0.533
158 26 Mean AUC : 0.543
232 18 Mean AUC : 0.553
306 13 Mean AUC : 0.545
380 11 Mean AUC : 0.559
454 9 Mean AUC : 0.563
528 7 Mean AUC : 0.566
602 6 Mean AUC : 0.581
676 6 Mean AUC : 0.588
750 5 Mean AUC : 0.575
824 5 Mean AUC : 0.583
898 4 Mean AUC : 0.610
972 4 Mean AUC : 0.592
1046 4 Mean AUC : 0.596
1120 3 Mean AUC : 0.604
1194 3 Mean AUC : 0.608
1268 3 Mean AUC : 0.609
1342 3 Mean AUC : 0.619
1416 2 Mean AUC : 0.613
1490 2 Mean AUC : 0.602
1500 2 Mean AUC : 0.616
processed/neo_poker_lab
8880
10 888 Mean AUC : 0.531
84 105 Mean AUC : 0.564
158 56 Mean AUC : 0.577
232 38 Mean AUC : 0.578
306 29 Mean AUC : 0.582
380 23 Mean AUC : 0.588
454 19 Mean AUC : 0.598
528 16 Mean AUC : 0.600
602 14 Mean AUC : 0.613
676 13 Mean AUC : 0.609
750 11 Mean AUC : 0.625
824 10 Mean AUC : 0.624
898 9 Mean AUC : 0.636
972 9 Mean AUC : 0.641
1046 8 Mean AUC : 0.637
1120 7 Mean AUC : 0.649
1194 7 Mean AUC : 0.653
1268 7 Mean AUC : 0.649
1342 6 Mean AUC : 0.662
1416 6 Mean AUC : 0.661
1490 5 Mean AUC : 0.673
1500 5 Mean AUC : 0.666
processed/tartanian5
6788
10 678 Mean AUC : 0.519
84 80 Mean AUC : 0.548
158 42 Mean AUC : 0.566
232 29 Mean AUC : 0.575
306 22 Mean AUC : 0.578
380 17 Mean AUC : 0.574
454 14 Mean AUC : 0.590
528 12 Mean AUC : 0.597
602 11 Mean AUC : 0.600
676 10 Mean AUC : 0.616
750 9 Mean AUC : 0.599
824 8 Mean AUC : 0.618
898 7 Mean AUC : 0.634
972 6 Mean AUC : 0.642
1046 6 Mean AUC : 0.641
1120 6 Mean AUC : 0.633
1194 5 Mean AUC : 0.647
1268 5 Mean AUC : 0.644
1342 5 Mean AUC : 0.656
1416 4 Mean AUC : 0.651
1490 4 Mean AUC : 0.670
1500 4 Mean AUC : 0.653
processed/sartre
1684
10 168 Mean AUC : 0.680
84 20 Mean AUC : 0.833
158 10 Mean AUC : 0.843
232 7 Mean AUC : 0.862
306 5 Mean AUC : 0.872
380 4 Mean AUC : 0.891
454 3 Mean AUC : 0.884
528 3 Mean AUC : 0.888
602 2 Mean AUC : 0.884
676 2 Mean AUC : 0.908
750 2 Mean AUC : 0.899
824 2 Mean AUC : 0.907
898 1 Mean AUC : 0.918
972 1 Mean AUC : 0.918
1046 1 Mean AUC : 0.927
1120 1 Mean AUC : 0.919
1194 1 Mean AUC : 0.915
1268 1 Mean AUC : 0.939
1342 1 Mean AUC : 0.931
1416 1 Mean AUC : 0.934
1490 1 Mean AUC : 0.940
1500 1 Mean AUC : 0.939
processed/lucky7_12
5940
10 594 Mean AUC : 0.520
84 70 Mean AUC : 0.602
158 37 Mean AUC : 0.603
232 25 Mean AUC : 0.644
306 19 Mean AUC : 0.638
380 15 Mean AUC : 0.652
454 13 Mean AUC : 0.665
528 11 Mean AUC : 0.669
602 9 Mean AUC : 0.667
676 8 Mean AUC : 0.666
750 7 Mean AUC : 0.686
824 7 Mean AUC : 0.680
898 6 Mean AUC : 0.689
972 6 Mean AUC : 0.688
1046 5 Mean AUC : 0.684
1120 5 Mean AUC : 0.699
1194 4 Mean AUC : 0.697
1268 4 Mean AUC : 0.706
1342 4 Mean AUC : 0.699
1416 4 Mean AUC : 0.715
1490 3 Mean AUC : 0.722
1500 3 Mean AUC : 0.705
processed/uni_mb_poker
6540
10 654 Mean AUC : 0.620
84 77 Mean AUC : 0.685
158 41 Mean AUC : 0.706
232 28 Mean AUC : 0.721
306 21 Mean AUC : 0.745
380 17 Mean AUC : 0.756
454 14 Mean AUC : 0.758
528 12 Mean AUC : 0.776
602 10 Mean AUC : 0.783
676 9 Mean AUC : 0.781
750 8 Mean AUC : 0.800
824 7 Mean AUC : 0.804
898 7 Mean AUC : 0.800
972 6 Mean AUC : 0.817
1046 6 Mean AUC : 0.824
1120 5 Mean AUC : 0.824
1194 5 Mean AUC : 0.830
1268 5 Mean AUC : 0.835
1342 4 Mean AUC : 0.850
1416 4 Mean AUC : 0.846
1490 4 Mean AUC : 0.857
1500 4 Mean AUC : 0.857
processed/all
84324
10 8432 Mean AUC : 0.554
84 1003 Mean AUC : 0.590
158 533 Mean AUC : 0.600
232 363 Mean AUC : 0.607
306 275 Mean AUC : 0.610
380 221 Mean AUC : 0.615
454 185 Mean AUC : 0.619
528 159 Mean AUC : 0.622
602 140 Mean AUC : 0.623
676 124 Mean AUC : 0.627
750 112 Mean AUC : 0.629
824 102 Mean AUC : 0.632
898 93 Mean AUC : 0.630
972 86 Mean AUC : 0.635
1046 80 Mean AUC : 0.634
1120 75 Mean AUC : 0.636
1194 70 Mean AUC : 0.640
1268 66 Mean AUC : 0.643
1342 62 Mean AUC : 0.640
1416 59 Mean AUC : 0.639
1490 56 Mean AUC : 0.646
1500 56 Mean AUC : 0.644
"""

smallConvLog = dict([(x.split('\n')[0],(x.split('\n')[1],[y.split(' ')[0] for y in x.split('\n')[2:-1]],[y.split(' ')[-1] for y in x.split('\n')[2:-1]])) for x in smallConvLog.split("processed/")])


def processHand(h, player, type, rest={}):
  if "STATE" not in h:
    return None

  categ = h.strip()[-1]
  h = h.strip()[:-1]

  phs = h.strip().split(":")

  hNo = int(phs[1])
  bets = phs[2].split("/")
  cards = phs[3].split("/")
  rez = phs[4].split("|")
  lastOrder=phs[5].split("|")
  firstOrder = [lastOrder[1],lastOrder[0]]

  cCards = cards[1:]
  if len(cCards) > 0:
    cCards[0] = [cCards[0][:2],cCards[0][2:4],cCards[0][4:6]]
  hCards = cards[0].split("|")
  hCards = [[hCards[0][:2],hCards[0][2:4]],[hCards[1][:2],hCards[1][2:4]]]
  curBet = [smallBlind, 2*smallBlind]
  curPlayer = 0
  nextPlayer = 1
  nbets = {}
  stages = ["pre-flop","flop","turn","river"]
  s = 0
  for bet in bets:
    curB = []
    i = 0
    while i < (len(bet)):
      if bet[i] == "f":
        curB.append({"betType":"f","betAmm":0})

      elif bet[i] == "c":
        diff = curBet[nextPlayer] - curBet[curPlayer]
        #print curPlayer, nextPlayer, diff
        curB.append({"betType":"c","betAmm":diff})
        curBet[curPlayer] += diff

      elif bet[i] == "r":
        nr = ""
        while bet[i+1] in "0123456789":
          i += 1
          nr += bet[i]
        nr = int(nr)

        diff = nr - curBet[curPlayer] - (curBet[nextPlayer] - curBet[curPlayer])
        curB.append({"betType":"r","betAmm":diff})
        curBet[curPlayer] += diff + (curBet[nextPlayer] - curBet[curPlayer])
      curPlayer = nextPlayer
      nextPlayer = (nextPlayer+1)%2
      i += 1
    nbets[stages[s]] = curB
    s += 1
  bets = nbets
  pot = {}
  curPot = 3*smallBlind
  toCall = smallBlind
  for s in range(len(bets)):
    stage = stages[s]
    pot[stage] = []
    for ind in range(len(bets[stage])):
      if bets[stage][ind]["betType"] == "c":
        curPot += toCall
        toCall = 0
      if bets[stage][ind]["betType"] == "r":
        curPot += toCall + bets[stage][ind]["betAmm"]
        toCall = bets[stage][ind]["betAmm"]
      pot[stage].append(curPot)

  if type == 0:
    if player == lastOrder[0]:
      pos = 0
      pot = pot["pre-flop"][-1]
    else:
      pos = 1
      pot = pot["flop"][0]
    raised = float(bets["flop"][pos]["betAmm"]) / pot
    return str(hNo)+","+str(pos)+","+str(pot)+","+str(raised)+","+categ
    
  elif type == 1:
    if player == lastOrder[0]:
      pos = 0
      cpot = pot["pre-flop"][-1]
    else:
      pos = 1
      cpot = pot["flop"][0]
    raised = float(bets["flop"][pos]["betAmm"]) / cpot
    noRaises = 0
    propRaised = []
    for i in range(len(bets["pre-flop"])):
      bp = firstOrder[i%2]
      if player == bp and bets["pre-flop"][i]["betType"] == "r":
        noRaises += 1
        propRaised.append(bets["pre-flop"][i]["betAmm"] / float(pot["pre-flop"][i-1]))
      
    avgRaised = 0
    if propRaised != []:
      avgRaised = sum(propRaised) / float(len(propRaised))

    avgEvals = trips[tuple(sorted(cCards[0]))]
    
    return str(hNo)+","+str(pos)+","+str(cpot)+","+str(raised)+","+str(noRaises)+","+str(avgRaised)+","+str(avgEvals)+","+categ

  elif type == 2:
    if "hNo" in rest and hNo < rest["hNo"]:
      rest = {}

    if player == lastOrder[0]:
      pos = 0
      cpot = pot["pre-flop"][-1]
      wonAmm = int(rez[0])
    else:
      pos = 1
      cpot = pot["flop"][0]
      wonAmm = int(rez[1])
    raised = float(bets["flop"][pos]["betAmm"]) / cpot
    noRaises = 0
    propRaised = []
    for i in range(len(bets["pre-flop"])):
      bp = firstOrder[i%2]
      if player == bp and bets["pre-flop"][i]["betType"] == "r":
        noRaises += 1
        propRaised.append(bets["pre-flop"][i]["betAmm"] / float(pot["pre-flop"][i-1]))
      
    avgRaised = 0
    if propRaised != []:
      avgRaised = sum(propRaised) / float(len(propRaised))

    avgEvals = trips[tuple(sorted(cCards[0]))] / float(10000000)
    
    rest["hNo"] = hNo

    try:
      rest["profitSoFar"] += max(wonAmm,0)
    except:
      rest["profitSoFar"] = max(wonAmm,0)

    try:
      rest["gambledSoFar"] += abs(wonAmm)
    except:
      rest["gambledSoFar"] = abs(wonAmm)

    try:
      rest["count"] += 1
    except:
      rest["count"] = 0

    if rest["gambledSoFar"] == 0:
      wonProp = 0
    else:
      wonProp = float(rest["profitSoFar"]) / rest["gambledSoFar"]

    return str(hNo)+","+str(pos)+","+str(cpot)+","+str(raised)+","+str(noRaises)+","+str(avgRaised)+","+str(avgEvals)+","+str(wonProp)+","+str(rest["count"])+","+categ, rest
    
    
def getBasicFeatures(fs):
  hands = []
  for fl in fs:
    player = fl.split('.')[0]
    with open(fl, 'r') as f:
      for line in f:
        hands.append(fl[10:] + "," + processHand(line,player,0))
  return hands
  
def getMedFeatures(fs):
  hands = []
  for fl in fs:
    player = fl.split('.')[0][10:]
    with open(fl, 'r') as f:
      for line in f:
        hands.append(fl[10:] + "," + processHand(line,player,1))
  return hands

def getAdvFeatures(fs):
  hands = []
  rest = {}
  for fl in fs:
    player = fl.split('.')[0][10:]
    with open(fl, 'r') as f:
      for line in f:
        rez, rest = processHand(line,player,2,rest)
        hands.append(fl[10:] + "," + rez)
  return hands
  
def getH2(prob):
  if prob == 1 or prob == 0:
    return 0
  rp = 1.0 - prob
  return -prob*math.log(prob,2) - rp*math.log(rp,2)
  
def testFeats(fns, cut, tp, fff):
  log = """processed/azure_sky
50 100 150 200 [0.00036636473994522412, 0.0017466223861320185, 0.03095812121536709, 0.10145587558417224, 2.4725510282652685e-05, 0.00034566549786510525, 0.00072847990643626348, 0.00028115982809956419]
processed/dcubot
50 100 150 200 [0.00077584926945556409, 0.0013147797344638668, 0.00015499588059009195, 0.0094233472168447907, 0.0007402901435439313, 0.00075668638156844992, 0.0023733632997249288, 0.0018181020777109924]
processed/hugh
50 100 150 200 [0.0011592695215058013, 0.11596269178316432, 0.0, 0.11713362126447124, 0.0, 0.11536624261884576, 0.0010025547498810905, 0.021263000127651477]
processed/hyperborean
50 100 150 200 [0.00043021380480745552, 0.00012427742788201357, 0.0086336496252388262, 0.0035269411704159648, 0.0036698951647499145, 0.0038115200143850059, 0.00049614646198237189, 0.019877934913555939]
processed/little_rock
50 100 150 200 [0.001308827044162908, 1.5770580331198047e-05, 0.0011096559232884884, 0.015342061575306598, 8.6493850488200685e-05, 0.00064878590582284623, 0.0012412145472696645, 0.0075139400876310658]
processed/spewy_louie
50 100 150 200 [0.00065948887640049358, 0.00010728650840169385, 0.0021457196328945605, 0.0029850158716374642, 0.00036205512549047913, 0.0012383575125949897, 0.0012002888113765153, 0.0042512078114010921]
processed/neo_poker_lab
50 100 150 200 [0.00069612853534428698, 0.041821152395642547, 0.0031195789289701903, 0.005891280336612903, 0.0035421739891592363, 0.046019268001859182, 0.0063380790459477421, 0.017548640818780936]
processed/tartanian5
50 100 150 200 [0.00070906750818489073, 0.0060729196119311268, 0.0016015854299765708, 0.0050950335251398471, 0.0035332677411563651, 0.0051044448001323905, 0.0014623889532907608, 0.010502773343874949]
processed/sartre
50 100 150 200 [0.0096147526506964454, 0.0077679286769583022, 0.16130951341024669, 0.098345601470345545, 0.04389392159699701, 0.20722374412559752, 0.0090277842149645293, 0.059037395893656752]
processed/lucky7_12
50 100 150 200 [0.00081885418944011024, 0.00056202571690749936, 0.001885538934680131, 0.022546009123236899, 0.001885538934680131, 0.0018933450001515362, 0.0015123904823309275, 0.0011122544961252867]
processed/uni_mb_poker
50 100 150 200 [0.0070572735905489736, 9.7975465399802175e-05, 0.027594372075557927, 0.058327672175541911, 0.069485370495971144, 0.069485370495971144, 0.15976249852220703, 0.01394470302364903]
processed/all
50 100 150 200 [0.00043939810940263335, 0.003223228582957649, 0.0056280612755822412, 0.016871458130461536, 0.0016475763866652393, 0.0016571154473158378, 0.00071392196094599569, 0.034905694184996561]

"""
  lcount = 0
  ldict = {}
  for line in log.strip().split('\n'):
    lcount += 1
    if lcount % 2 == 1:
      name = line.strip()
    else:
      ldict[name] = map(float,line.strip().split('[')[-1].strip()[:-1].split(", "))

  count = -1
  size = (2,6)
  fig = plt.figure(figsize=(size[1]*2.5,size[0]*2.5))
  gs = gridspec.GridSpec(size[0], size[1])
  ylim = [(-200,3200),(-1,2),(100,100000),(0.001,1000),(-0.5,4.5),(-0.5,10.5),(1.2,1.5),(-0.1,1.1),(-200,3200)]

  minFeats = {}
  storedMinFeats = {'processed/tartanian5-adv': [7, 1, 5, 3, 4, 2, 6, 0], 'processed/hugh-adv': [3, 1, 5, 7, 0, 6, 4, 2], 'processed/neo_poker_lab-adv': [5, 1, 7, 6, 3, 4, 2, 0], 'processed/little_rock-adv': [3, 7, 0, 6, 2, 5, 4, 1], 'processed/hyperborean-adv': [7, 2, 5, 4, 3, 6, 0, 1], 'processed/uni_mb_poker-adv': [6, 5, 4, 3, 2, 7, 0, 1], 'processed/azure_sky-adv': [3, 2, 1, 6, 0, 5, 7, 4], 'processed/sartre-adv': [5, 2, 3, 7, 4, 0, 6, 1], 'processed/all-adv': [7, 3, 2, 1, 5, 4, 6, 0], 'processed/dcubot-adv': [3, 6, 7, 1, 0, 5, 4, 2], 'processed/lucky7_12-adv': [3, 5, 4, 2, 6, 7, 0, 1], 'processed/spewy_louie-adv': [7, 3, 2, 5, 6, 0, 4, 1]}

  for fn in fns:
    count += 1
    feats = []
    with open(fn,'r') as f:
      lineNo = 0
      for line in f:
        lineNo += 1
        #if lineNo == 100:
        #  break
        feats.append(map(float,line.strip().split(',')[1:]))
    ones = filter(lambda x: x[-1] == 1.0, feats)
    zeros = filter(lambda x: x[-1] == 0.0, feats)
    #ones = random.sample(ones,len(zeros))
    #feats = ones + zeros
    #random.shuffle(feats)
    print fn[:-cut]

    #dot_data = StringIO.StringIO() 
    #tree.export_graphviz(clf, out_file="imgs\\" + fn[:-6]+".dot") 
    #graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    #graph.write_png("imgs\\" + fn[:-6]+".png") 

    row = count/size[1]
    col = count % size[1]    
    ax = fig.add_subplot(gs[row, col])
    if tp == 0:
      o = float(len(ones))
      t = len(feats)
      h2 = getH2(o/t)

      divIters = 2000
      igs = []
      known = []#[0,1,2,3,4,5,6,7]
      for i in range(8):
        if i in known:
          ig = ldict[fn[:-cut]][i]
        elif i == 1:
          div = 0.5
          fstPos = filter(lambda x: x[i] < div, feats)
          sndPos = filter(lambda x: x[i] >= div, feats)
          if len(fstPos) == 0 or len(sndPos) == 0:
            ig = 0
          else:
            fstPosOnes = float(len(filter(lambda x: x[-1] == 1.0, fstPos)))
            sndPosOnes = float(len(filter(lambda x: x[-1] == 1.0, sndPos)))
            fstPosH2 = getH2(fstPosOnes/len(fstPos))
            sndPosH2 = getH2(sndPosOnes/len(sndPos))
            probFst = float(len(fstPos)) / len(feats)
            probSnd = 1-probFst
            nh2 = probFst*fstPosH2 + probSnd*sndPosH2
            ig = h2 - nh2
        else:
          featOnes = [x[i] for x in ones]
          featZeros = [x[i] for x in zeros]
          q1Ones =  scoreatpercentile(featOnes,10)
          q3Ones =  scoreatpercentile(featOnes,90)
          q1Zeros =  scoreatpercentile(featZeros,10)
          q3Zeros =  scoreatpercentile(featZeros,90)
          
          start = min(q1Ones, q1Zeros)
          end = max(q3Ones, q3Zeros)
          inc = (end-start) / float(divIters)
          if inc == 0:
            inc = 1
          div = None
          ig = -1
          ccc = 0
          while ccc < divIters:
            ccc += 1
            if ccc % 50 == 0:
              print ccc,
              sys.stdout.flush()

            div = random.random() * (end-start) + start
            fstPos = filter(lambda x: x[i] < div, feats)
            sndPos = filter(lambda x: x[i] >= div, feats)
            if len(fstPos) == 0 or len(sndPos) == 0:
              cig = 0
            else:
              fstPosOnes = float(len(filter(lambda x: x[-1] == 1.0, fstPos)))
              sndPosOnes = float(len(filter(lambda x: x[-1] == 1.0, sndPos)))    
              fstPosH2 = getH2(fstPosOnes/len(fstPos))
              sndPosH2 = getH2(sndPosOnes/len(sndPos))
              probFst = float(len(fstPos)) / len(feats)
              probSnd = 1-probFst
              nh2 = probFst*fstPosH2 + probSnd*sndPosH2
              cig = h2 - nh2
            #print i, start, end, cig
            if cig > ig:
              ig = cig
              div = start

            #start += inc
        #print i, div, ig
        igs.append(ig)
        #print fstPosOnes, len(fstPos), fstPosH2, sndPosOnes, len(sndPos), sndPosH2, probFst*fstPosH2 + probSnd*sndPosH2
      print igs
      ax.bar(0,0.7,width=igs,bottom=[0,1,2,3,4,5,6,7], orientation="horizontal", alpha = 0.3)
      ax.set_yticks([0.3,1.3,2.3,3.3,4.3,5.3,6.3,7.3])
      ax.set_yticklabels(["No. of hands played","Position","Pot amount", "Raise / Pot", "No. of pre-flop raises", "Avg. pre-flop raise", "Flop strength", "Prop. sim. hands won"], ha="left", x=0.05)
      plt.xlim(0,0.22)
      plt.ylim(-0.2,8)
      plt.title(fn[10:-cut], fontsize=20)
      plt.suptitle("Information gain", fontsize=30)

      minFeats[fn] = [x[1] for x in sorted(zip(igs,range(8)), reverse=True)]
        
    elif tp == 1:
      #print median([x[fff] for x in ones]), median([x[fff] for x in ones])
      ax.boxplot([[x[fff] for x in ones],[x[fff] for x in zeros]],whis=20)
      ax.set_xticklabels(["Bluff", "Non-bluff"], fontsize=15)
      plt.ylim(ylim[fff][0],ylim[fff][1])
      plt.title(fn[10:-cut],  fontsize=20)
      if fff in [2,3]:
        ax.set_yscale('log')
      titles = ["Number of hands played","Position","Pot amount", "Raise / Pot", "Number of pre-flop raises", "Average pre-flop raise / pot", "Flop strength", "Proportion won in similar hands","hh"]
      plt.suptitle(titles[fff], fontsize=30)

    elif tp == 2:
      noFeats = 8
      #print len(ones), len(zeros), len(feats)
      
      if len(zeros) <= len(ones):
        feats = zeros + random.sample(ones,len(zeros))
      else:
        feats = ones + random.sample(zeros,len(ones))
      random.shuffle(feats)
      print len(feats), storedMinFeats[fn][:noFeats]


      X = []
      Y = []
      for f in feats:
        X.append([f[i] for i in storedMinFeats[fn][:noFeats]])
        if f[-1] == 1.0:
          Y.append(1)
        else:
          Y.append(-1)

      cv = StratifiedKFold(Y, n_folds=10)
      mean_tpr = 0.0
      mean_fpr = np.linspace(0, 1, 100)

      for i, (train, test) in enumerate(cv):
        #print set(train).intersection(set(test))
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        probas_ = clf.fit([X[i] for i in train], [Y[i] for i in train]).predict_proba([X[i] for i in test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = metrics.roc_curve([Y[i] for i in test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        auc = metrics.auc(fpr, tpr)
        print '%0.2f' % auc,
        sys.stdout.flush()

      mean_tpr /= len(cv)
      mean_tpr[-1] = 1.0
      mean_auc = metrics.auc(mean_fpr, mean_tpr)

      print "Mean AUC : %0.3f" % mean_auc
      row = count/size[1]
      col = count % size[1]
      ax = fig.add_subplot(gs[row, col])

      ax.plot(mean_fpr, mean_tpr, label='AUC=%0.2f' % mean_auc)
      ax.plot([0, 1], [0, 1], 'k--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.0])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(fn[10:-cut], fontsize=20)
      plt.legend(loc="lower right", fontsize=12)
      fig.suptitle("ROC Curves", fontsize=30)

    elif tp == 3:
      noFeats = 8
      which = 0
      
      if len(zeros) <= len(ones):
        feats = zeros + random.sample(ones,len(zeros))
      else:
        feats = ones + random.sample(zeros,len(ones))
      random.shuffle(feats)
      print len(feats)

      iters = 20
      if which == 0:
        curSize = 10
        maxSize = 1500
      else:
        curSize = float(len(feats)) / 70
        maxSize = float(len(feats))
      inc = (maxSize - curSize) / iters
      aucs = []
      sizes = []
      props = []
      last = False
      while curSize <= maxSize or not last:
        if curSize > maxSize:
          curSize = maxSize
          last = True
        reps = int(len(feats) / curSize)
        meanMeanAuc = 0
        mX = []
        mY = []
        for f in feats:
          mX.append([f[i] for i in storedMinFeats[fn][:noFeats]])
          if f[-1] == 1.0:
            mY.append(1)
          else:
            mY.append(-1)

        mcv = StratifiedKFold(mY, n_folds=reps)
        print curSize, reps,

        for i, (mtrain, mtest) in enumerate(mcv):
          rtest = random.sample(mtest,int(curSize))
          X = [mX[i] for i in rtest]
          Y = [mY[i] for i in rtest]
          clf = tree.DecisionTreeClassifier()
          cv = StratifiedKFold(Y, n_folds=3)
          meanAuc = 0
          lenTrain = 0
          for i, (train, test) in enumerate(cv):
            lenTrain += len(train)
            probas_ = clf.fit([X[i] for i in train], [Y[i] for i in train]).predict_proba([X[i] for i in test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve([Y[i] for i in test], probas_[:, 1])
            auc = metrics.auc(fpr, tpr)
            #print '%0.2f' % auc,
            sys.stdout.flush()
            meanAuc += auc

          lenTrain /= len(cv)
          meanAuc /= len(cv)
          #print "%0.2f" % meanAuc,
          meanMeanAuc += meanAuc
        meanMeanAuc /= reps
        aucs.append(meanMeanAuc)
        props.append(float(curSize) / maxSize)
        sizes.append(lenTrain)
        print "Mean AUC : %0.3f" % meanMeanAuc
        curSize += inc
      
      if which == 0:
        ax.plot(sizes, aucs)
        ax.set_xlim([0, 150.0])
        plt.xticks([0, 50, 100, 150])
        plt.xlabel('Ammount training data')
      else:
        ax.plot(props, aucs)
        ax.set_xlim([0, 1.0])
        plt.xlabel('Proportion data used')

      ax.set_ylim([0.0, 1.0])
      plt.ylabel('Mean AUC')
      plt.title(fn[10:-cut])
      fig.suptitle("Convergence rate of AUC", fontsize=30)

    elif tp == 4:
      X = []
      Y = []
      for f in feats:
        X.append([f[5]])
        if f[-1] == 1.0:
          Y.append(1)
        else:
          Y.append(-1)

      clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=2)
      clf = clf.fit(X, Y)
      #print clf.get_params()
      
      dot_data = StringIO.StringIO() 
      tree.export_graphviz(clf, out_file=dot_data) 
      graph = pydot.graph_from_dot_data(dot_data.getvalue())
      pdf = fn[10:-cut] + "-tree.pdf"
      graph.write_pdf(pdf)

    elif tp == 5:
      o = float(len(ones))
      t = len(feats)
      h2 = getH2(o/t)
      div = 13718209.5
      fstPos = filter(lambda x: x[5] < div, feats)
      sndPos = filter(lambda x: x[5] >= div, feats)
      if len(fstPos) == 0 or len(sndPos) == 0:
        ig = 0
      else:
        fstPosOnes = float(len(filter(lambda x: x[-1] == 1.0, fstPos)))
        sndPosOnes = float(len(filter(lambda x: x[-1] == 1.0, sndPos)))
        fstPosH2 = getH2(fstPosOnes/len(fstPos))
        sndPosH2 = getH2(sndPosOnes/len(sndPos))
        probFst = float(len(fstPos)) / len(feats)
        probSnd = 1-probFst
        nh2 = probFst*fstPosH2 + probSnd*sndPosH2
        ig = h2 - nh2
      print ig

    elif tp == 6:
      curLog = convLog[fn[10:-cut]]
      props = [float(curLog[1][i])/float(curLog[0]) for i in range(len(curLog[1]))]
      ax.plot(props, map(float,curLog[2]))
      ax.set_xlim([0, 1.0])
      plt.xlabel('Proportion data used')

      ax.set_ylim([0.0, 1.0])
      plt.ylabel('Mean AUC')
      plt.title(fn[10:-cut], fontsize=20)
      fig.suptitle("Convergence rate of AUC", fontsize=30)

    elif tp == 7:

      finAuc = {"azure_sky":0.96,"dcubot":0.85,"hugh":0.92,"hyperborean":0.86,"little_rock":0.85,"spewy_louie":0.84,"neo_poker_lab":0.93, "tartanian5":0.9, "sartre":0.97, "lucky7_12":0.9, "uni_mb_poker":0.99, "all":0.91}

      curLog = smallConvLog[fn[10:-cut]]
      props = [float(curLog[1][i])*0.66666 for i in range(len(curLog[1]))]
      ax.plot( props, map(float,curLog[2]))
      ax.set_xlim([0, 1000.0])
      plt.xticks([0, 500, 1000, 1000])
      plt.xlabel('Amount training data')
      ax.plot([0,1000],[finAuc[fn[10:-cut]],finAuc[fn[10:-cut]]], 'r', linewidth=2)
      ax.set_ylim([0.0, 1.02])
      plt.ylabel('Mean AUC')
      plt.title(fn[10:-cut], fontsize=20)
      fig.suptitle("Convergence rate of AUC", fontsize=30)

  if tp in [0,1,2,3,6,7]:
    gs.tight_layout(fig, rect=[0, 0, 1, 0.9], pad=0.5)
    plt.show()

  if tp == 0:
    print minFeats
  
def getCardTriples():
  trips = {}
  deck = sorted(pe.card2string(pe.deck()))
  for i1 in range(len(deck)):
    print i1
    for i2 in range(i1+1,len(deck)):
      for i3 in range(i2+1, len(deck)):
        evals = []
        for i4 in range(len(deck)):
          for i5 in range(len(deck)):
            evals.append(pe.evaln([deck[i1],deck[i2],deck[i3],deck[i4],deck[i5]]))
        if len(evals) > 0:
          trips[(deck[i1],deck[i2],deck[i3])] = sum(evals) / float(len(evals))
  return trips

if __name__ == "__main__":
  fdir="processed/"
  if sys.argv[1] == "0":
    allHands = []
    players = ["azure_sky","dcubot","hugh","hyperborean","little_rock","spewy_louie","neo_poker_lab","tartanian5","sartre","lucky7_12","uni_mb_poker"]
    for player in players:
      fs = filter(lambda x: x[:len(player)+1] == player+".", os.listdir('.'))
      print player, len(fs)
      fn = player + "-basic"
      hands = getBasicFeatures(fs)
      allHands += hands
      with open(fn,'w') as f:
        f.write('\n'.join(hands))
    with open("all-basic",'w') as f:
      f.write('\n'.join(allHands))
  if sys.argv[1] == "1":
    allHands = []
    players =["azure_sky","dcubot","hugh","hyperborean","little_rock","spewy_louie","neo_poker_lab","tartanian5","sartre","lucky7_12","uni_mb_poker"]
    for player in players:
      fs = [fdir + x for x in filter(lambda x: x[:len(player)+1] == player+".", os.listdir(fdir))]
      #print fs
      print player, len(fs)
      fn = fdir + player + "-adv"
      hands = getAdvFeatures(fs)
      allHands += hands
      with open(fn,'w') as f:
        f.write('\n'.join(hands))
    with open(fdir + "all-adv",'w') as f:
      f.write('\n'.join(allHands))
  elif sys.argv[1] == "2":
    tInd = 2
    ts = ["-basic","-med","-adv"]
    files = [fdir + x + ts[tInd] for x in ["azure_sky","dcubot","hugh","hyperborean","little_rock","spewy_louie","neo_poker_lab","tartanian5","sartre","lucky7_12","uni_mb_poker", "all"]]
    try:
      argVar = int(sys.argv[3])
    except:
      argVar = None

    testFeats(files, len(ts[tInd]), int(sys.argv[2]), argVar)
  elif sys.argv[1] == "3":
    trips = getCardTriples()
    with open("tripHandVals", 'w') as f:
      pickle.dump(trips,f)
  else:
    print "Invalid argument"
