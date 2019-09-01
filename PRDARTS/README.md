# Prune and Replace NAS

## Generate a Random Architecture

```
morph_restrictions='r_depth'
if morph_restrictions=='darts_like':
    conv_min_k=3
    conv_max_k=7
    conv_max_dil=2
    conv_max_c_mul_len=1
    conv_max_c_mul_num=1
    conv_max_c_mul_width=1
elif morph_restrictions=='unrestricted':
    conv_min_k=1
    conv_max_k=7
    conv_max_dil=2
    conv_max_c_mul_len=999
    conv_max_c_mul_num=999
    conv_max_c_mul_width=999
elif args.morph_restrictions=='r_depth':
    conv_min_k = 1
    conv_max_k = 7
    conv_max_dil = 2
    conv_max_c_mul_len = 1
    conv_max_c_mul_num = 5
    conv_max_c_mul_width = 5
c_muls = [[]]
for l in range(1, conv_max_c_mul_width + 1): #To be corrected for unrestricted search space
    cmul = []
    cmul.append(l)
    c_muls.append(cmul)
CONV_PRIMITIVES = [('conv', {'dil': i, 'c_mul': cmul, 'k': k}) for i in range(1, conv_max_dil + 1) for k in
                   range(conv_min_k, conv_max_k + 1) for cmul in c_muls]
PRIMITIVES = [('skip', {}), ('pool', {'k': 3, 'type_': 'max'}),
              ('pool', {'k': 3, 'type_': 'avg'})] + CONV_PRIMITIVES
from genotypes import Genotype
import random

normal = []
reduce = []
ops_normal = {}
ops_reduce = {}
links_normal = {}
links_reduce = {}
for i in range(4):
    #Giving fixed chances for skip and pooling operations as PRDARTS never searches for all convolutions of the space at the same time
    ops_normal[i] = random.choices([j for j in range(len(PRIMITIVES))],
                                   weights=[1 / 8, 1 / 8, 1 / 8] + len(CONV_PRIMITIVES) * [
                                   1 / len(CONV_PRIMITIVES)], k=2)
    ops_reduce[i] = random.choices([j for j in range(len(PRIMITIVES))],
                                   weights=[1 / 8, 1 / 8, 1 / 8] + len(CONV_PRIMITIVES) * [
                                   1 / len(CONV_PRIMITIVES)], k=2)
    links_normal[i] = random.sample([j for j in range(i + 2)], 2)
    links_reduce[i] = random.sample([j for j in range(i + 2)], 2)
    normal.append((PRIMITIVES[ops_normal[i][0]][0], PRIMITIVES[ops_normal[i][0]][1], links_normal[i][0]))
    reduce.append((PRIMITIVES[ops_reduce[i][0]][0], PRIMITIVES[ops_reduce[i][0]][1], links_reduce[i][0]))
    normal.append((PRIMITIVES[ops_normal[i][1]][0], PRIMITIVES[ops_normal[i][1]][1], links_normal[i][1]))
    reduce.append((PRIMITIVES[ops_reduce[i][1]][0], PRIMITIVES[ops_reduce[i][1]][1], links_reduce[i][1]))
reduce_concat = [2, 3, 4, 5]
normal_concat = [2, 3, 4, 5]
genotype = Genotype(normal, normal_concat, reduce, reduce_concat)
```

## Search

```
python train_search.py 
--dataset CIFAR10 # choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102
--tmp_data_dir /data # path to your data
--save test 
--batch_size_min 24 
--num_morphs "3,   3,   3,   3,   3,   0,   0,   0,   0" 
--grace_epochs "5,   5,   3,   3,   3,   3,   3,   3,   3" 
--num_to_keep="6,   6,   6,   6,   6,   4,   3,   2,   1" 
--epochs "15,  15,  10,  10,  10,  10,  10,  10,  10" 
--try_load "true" 
--report_freq 50 
--test "false" 
--batch_multiples 8 
--batch_size_max 128 
--primitives 'prdarts4'
--dropout_rate="0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0" 
--add_layers "3,   3,   3,   3,   3,   3,   3,   3,   3" 
--init_channels 16 
--add_width "0,   0,   0,   0,   0,   0,   0,   0,   0" 
--seed 0 
--morph_restrictions 'r_depth' 
--learning_rate_min 0.01
```

## Augment

```
python train_cifar.py 
--dataset CIFAR10 # choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102 
--tmp_data_dir /data # path to your data 
--save test 
--learning_rate 0.025 
--batch_size_min 32 
--epochs 600 
--auxiliary "true" 
--batch_size_max=128 
--workers 4 
--batch_multiples 8 
--seed 0 
--arch genotype 
--drop_path_prob 0.3 
--cutout "true"
--init_channels 36 
--layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102
```
