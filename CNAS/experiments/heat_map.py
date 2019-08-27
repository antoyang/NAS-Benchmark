import time
import numpy as np
import seaborn as sns
from pandas import Series,DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from darts.utils import create_exp_dir

class HeapMap(object):
    def __init__(self):
        super(HeapMap, self).__init__()
        self.store_path = './heap_map/' + 'graph-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M%S'))
        create_exp_dir(self.store_path, scripts_to_save=None)
        self.normal_array = \
        np.array([[0.1718, 0.1675, 0.0404, 0.0415, 0.1407, 0.1185, 0.3196],
        [0.2320, 0.2777, 0.0293, 0.0365, 0.1056, 0.1144, 0.2045],
        [0.2790, 0.2823, 0.0455, 0.0483, 0.0959, 0.0934, 0.1557],
        [0.2479, 0.2862, 0.0348, 0.0425, 0.0799, 0.1855, 0.1232],
        [0.1644, 0.4227, 0.0221, 0.0322, 0.0543, 0.1427, 0.1614],
        [0.2697, 0.2799, 0.0431, 0.0439, 0.1305, 0.0710, 0.1620],
        [0.2961, 0.3373, 0.0399, 0.0464, 0.0785, 0.0802, 0.1217],
        [0.1484, 0.4467, 0.0248, 0.0343, 0.0789, 0.0912, 0.1757],
        [0.0534, 0.7706, 0.0154, 0.0177, 0.0249, 0.0454, 0.0725],
        [0.2279, 0.3509, 0.0416, 0.0399, 0.0992, 0.1016, 0.1390],
        [0.2491, 0.3712, 0.0404, 0.0452, 0.0738, 0.0917, 0.1287],
        [0.1132, 0.5628, 0.0213, 0.0275, 0.0497, 0.0732, 0.1523],
        [0.0419, 0.8060, 0.0144, 0.0159, 0.0294, 0.0421, 0.0502],
        [0.0261, 0.8249, 0.0162, 0.0183, 0.0343, 0.0375, 0.0427]])

        self.reduce_array = \
        np.array([[0.0878, 0.1235, 0.1910, 0.2339, 0.1080, 0.1146, 0.1412],
        [0.1000, 0.1843, 0.1368, 0.1768, 0.1326, 0.1264, 0.1431],
        [0.0889, 0.1127, 0.1696, 0.2059, 0.1317, 0.1106, 0.1805],
        [0.1395, 0.1231, 0.1485, 0.1959, 0.1331, 0.1359, 0.1239],
        [0.1732, 0.1628, 0.0814, 0.0900, 0.1416, 0.1546, 0.1965],
        [0.1283, 0.1254, 0.1641, 0.1768, 0.1200, 0.1361, 0.1493],
        [0.1115, 0.1263, 0.1467, 0.1745, 0.1582, 0.1249, 0.1579],
        [0.1609, 0.1568, 0.0761, 0.0773, 0.1792, 0.1443, 0.2055],
        [0.1413, 0.1592, 0.0743, 0.0706, 0.1569, 0.2406, 0.1571],
        [0.1275, 0.1215, 0.1779, 0.1909, 0.1146, 0.1279, 0.1397],
        [0.1250, 0.1513, 0.1504, 0.1723, 0.1298, 0.1243, 0.1469],
        [0.1539, 0.1432, 0.0772, 0.0717, 0.1476, 0.1760, 0.2305],
        [0.1856, 0.2151, 0.0833, 0.0713, 0.1028, 0.1746, 0.1674],
        [0.1552, 0.2100, 0.0764, 0.0677, 0.1766, 0.1507, 0.1634]])
        index = ['n(0,0)', 'n(0,1)',
                 'n(1,0)', 'n(1,1)', 'n(1,2)',
                 'n(2,0)', 'n(2,1)', 'n(2,2)', 'n(2,3)',
                 'n(3,0)', 'n(3,1)', 'n(3,2)', 'n(3,3)', 'n(3,4)']

        OPs = [ 'skip_connect','cweight_com','avg_pool_3x3',
                'max_pool_3x3','sep_conv_3x3','dil_conv_3x3', 'shuffle_conv_3x3',]
        self.df1 = DataFrame(self.normal_array, index=index, columns=OPs)
        self.df2 = DataFrame(self.reduce_array, index=index, columns=OPs)


    def draw(self):
        f, ax1= plt.subplots(figsize=(15, 9))
        sns.heatmap(self.df1, annot=True, ax=ax1,
                    annot_kws={'size': 13, 'weight': 'bold'})
        ax1.set_xlabel('Ops without none operation', labelpad=14, fontsize='medium')
        ax1.set_ylabel('Possiable Input Index', labelpad=14, fontsize='medium')
        # ax1.set_title('The weights for Ops without none operation in normal cell', pad = 18, fontsize='x-large')

        # f, ax2= plt.subplots(figsize=(15, 9))
        # sns.heatmap(self.df2, annot=True, ax=ax2,
        #             annot_kws={'size': 13, 'weight': 'bold'})
        # ax2.set_xlabel('Ops without none operation', labelpad=14, fontsize='medium')
        # ax2.set_ylabel('Possible predecessors id for each intermediate node', labelpad=14, fontsize='medium')
        # #ax2.set_title('The weights for Ops without none operation in reduction cell', pad = 18, fontsize='x-large')
        plt.savefig(self.store_path+'/normal_hm.pdf', bbox_inches = 'tight', dpi=600)
        # plt.show()


if __name__ == '__main__':
    hm = HeapMap()
    hm.draw()

