from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
                    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

#-----------------------Train in cifar 10
'''
1. Not consider none in genotypes
'''
# meta-node = 3; one step
DARTS_MORE_V1 = Genotype(normal=[('cweight_com', 1), ('sep_conv_3x3', 0), ('cweight_com', 0), ('cweight_com', 1), ('cweight_com', 0), ('cweight_com', 1)], normal_concat=range(2, 5),
        reduce=[('cweight_com', 0), ('shuffle_conv_3x3', 1), ('shuffle_conv_3x3', 2), ('max_pool_3x3', 0), ('shuffle_conv_3x3', 3), ('shuffle_conv_3x3', 2)], reduce_concat=range(2, 5))

# meta-node = 4; one step
DARTS_MORE_V2 = Genotype(normal=[('shuffle_conv_3x3', 0), ('cweight_com', 1), ('cweight_com', 1), ('cweight_com', 0), ('cweight_com', 1), ('cweight_com', 0), ('cweight_com', 0), ('cweight_com', 1)], normal_concat=range(2, 6),
        reduce=[('cweight_com', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('cweight_com', 1), ('avg_pool_3x3', 0), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 3)], reduce_concat=range(2, 6))

# meta-node = 5; one step
DARTS_MORE_V3 = Genotype(normal=[('shuffle_conv_3x3', 0), ('cweight_com', 1), ('cweight_com', 2), ('dil_conv_3x3', 0), ('cweight_com', 2), ('skip_connect', 0), ('cweight_com', 1), ('cweight_com', 0), ('cweight_com', 0), ('cweight_com', 1)], normal_concat=range(2, 7),
        reduce=[('max_pool_3x3', 0), ('shuffle_conv_3x3', 1), ('max_pool_3x3', 0), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 3), ('shuffle_conv_3x3', 0), ('dil_conv_3x3', 3), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 4), ('shuffle_conv_3x3', 2)], reduce_concat=range(2, 7))

# meta-node = 6; one step
DARTS_MORE_V4 = Genotype(
    normal=[('sep_conv_3x3', 1), ('shuffle_conv_3x3', 0), ('skip_connect', 0), ('cweight_com', 1), ('cweight_com', 3),
            ('cweight_com', 1), ('cweight_com', 4), ('cweight_com', 3), ('cweight_com', 3), ('cweight_com', 4),
            ('cweight_com', 4), ('cweight_com', 3)], normal_concat=range(2, 8),
    reduce=[('max_pool_3x3', 1), ('cweight_com', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2),
            ('max_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 4), ('skip_connect', 3),
            ('skip_connect', 4), ('skip_connect', 3)], reduce_concat=range(2, 8))


"""
2. Consider none in genotypes
"""

# meta-node 3; one step
DARTS_MORE_NONE_V1 = Genotype(normal=[('shuffle_conv_3x3', 1), ('sep_conv_3x3', 0), ('none', 2), ('none', 1), ('none', 3), ('none', 2)], normal_concat=range(2, 5),
                    reduce=[('max_pool_3x3', 0), ('shuffle_conv_3x3', 1), ('max_pool_3x3', 0), ('shuffle_conv_3x3', 2), ('dil_conv_3x3', 3), ('cweight_com', 0)], reduce_concat=range(2, 5))

DARTS_MORE_NONE_V2 = Genotype(normal=[('cweight_com', 1), ('sep_conv_3x3', 0), ('none', 2), ('cweight_com', 1), ('none', 3), ('none', 2), ('none', 4), ('none', 3)], normal_concat=range(2, 6),
                    reduce=[('shuffle_conv_3x3', 1), ('shuffle_conv_3x3', 0), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('none', 4), ('shuffle_conv_3x3', 3)], reduce_concat=range(2, 6))

# meta-node 5; one step
DARTS_MORE_NONE_V3 = Genotype(normal=[('shuffle_conv_3x3', 0), ('none', 1), ('none', 2), ('shuffle_conv_3x3', 0), ('none', 3), ('none', 2),('none', 4), ('none', 3), ('none', 5), ('none', 4)], normal_concat=range(2, 7),
        reduce=[('cweight_com', 1), ('shuffle_conv_3x3', 0), ('shuffle_conv_3x3', 2), ('cweight_com', 1), ('cweight_com', 1),('shuffle_conv_3x3', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('shuffle_conv_3x3', 1), ('none', 5)],reduce_concat=range(2, 7))

# meta-node 6; one step
DARTS_MORE_NONE_V4 = Genotype(normal=[('sep_conv_3x3', 0), ('none', 1), ('none', 2), ('shuffle_conv_3x3', 0), ('none', 3), ('none', 2), ('none', 4), ('none', 3), ('none', 5), ('none', 4), ('none', 5), ('none', 6)], normal_concat=range(2, 8),
         reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 2), ('avg_pool_3x3', 1), ('cweight_com', 4), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 2), ('shuffle_conv_3x3', 4), ('none', 5), ('none', 6)], reduce_concat=range(2, 8))

#-----------------------Train on tiny-imagenet

# meta-node 4; one step
DARTS_MORE_TY2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('shuffle_conv_3x3', 2), ('max_pool_3x3', 0), ('cweight_com', 1), ('sep_conv_3x3', 3), ('cweight_com', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('cweight_com', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('cweight_com', 1), ('cweight_com', 0), ('shuffle_conv_3x3', 3)], reduce_concat=range(2, 6))


# test
DARTS = DARTS_MORE_V1
