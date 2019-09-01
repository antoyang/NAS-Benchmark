# PC-DARTS

## Generate a Random Architecture

```
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype

import copy
import random
import torch.nn.functional as F

n_ops = 8
n_nodes = 4
S = 0
for i in range(4):
    S = S + 2 + i

switches = []
for i in range(S):
    switches.append([True for j in range(n_ops)])
switches_normal = copy.deepcopy(switches)
switches_reduce = copy.deepcopy(switches)
for i in range(S):
    # excluding zero operations
    switches_normal[i][0] = False
    switches_reduce[i][0] = False
    idxs = [1 + i for i in range(n_ops - 1)]
    # randomly 6 dropping operations out of the 7 possible
    drop_normal = random.sample(idxs, n_ops - 2)
    drop_reduce = random.sample(idxs, n_ops - 2)
    for idx in drop_normal:
        switches_normal[i][idx] = False
    for idx in drop_normal:
        switches_reduce[i][idx] = False
model = Network(16, 10, 20, None)
model = model.cuda()

# Generate architecture
sm_dim = -1
arch_param = model.arch_parameters()
normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
normal_final = [0 for idx in range(S)]
reduce_final = [0 for idx in range(S)]
keep_normal = [0, 1]
keep_reduce = [0, 1]
n = 3
start = 2
for i in range(3):
    end = start + n
    tbsn = normal_final[start:end]
    tbsr = reduce_final[start:end]
    edge_n = random.sample([k for k in range(n)], 2)
    keep_normal.append(edge_n[-1] + start)
    keep_normal.append(edge_n[-2] + start)
    edge_r = random.sample([k for k in range(n)], 2)
    keep_reduce.append(edge_r[-1] + start)
    keep_reduce.append(edge_r[-2] + start)
    start = end
    n = n + 1
for i in range(S):
    if not i in keep_normal:
        for j in range(n_ops):
            switches_normal[i][j] = False
    if not i in keep_reduce:
        for j in range(n_ops):
            switches_reduce[i][j] = False


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene

    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype

genotype = parse_network(switches_normal, switches_reduce)
```

## Search

```
python train_search.py 
--dataset CIFAR10 # choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102
--datapath /data # path to your data 
--save test
```

# Augment

```
python train.py 
--dataset CIFAR10 # choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102
--datapath /data # path to your data 
--save test  
--auxiliary 
--cutout 
--arch genotype
--layers 20 #20 for CIFAR datasets, 8 for Sport8, MIT67 and flowers102
