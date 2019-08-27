# ENAS CIFAR-10 Implementation in PyTorch
"Efficient Neural Architecture Search via Parameter Sharing" 

Used for experiments on Sport8 and MIT67

## Requirements

PyTorch 0.4.0+

## Generate a random architecture

import numpy.random as rd
B = 5
ops = rd.randint(0, 5, 4*B) #5 ops, B nodes
links = rd.choice([0, 1], size=4*B, replace=True)
arch_normal = [links[0], ops[0], links[1], ops[1], links[0], ops[2], links[1], ops[3], links[0], ops[4], links[1], ops[5],
links[0], ops[6], links[1], ops[7], links[0], ops[16], links[1], ops[17]]
arch_reduce = [links[0], ops[8], links[1], ops[9], links[0], ops[10], links[1], ops[11], links[0], ops[12], links[1], ops[13],
links[0], ops[14], links[1], ops[15], links[0], ops[18], links[1], ops[19]]
op = {0:"sep_conv_3x3", 1:"sep_conv_5x5", 2:"avg_pool_3x3", 3:"max_pool_3x3", 4:"skip_connect"}
genotype = {}
genotype["normal"]=[]
genotype["reduce"]=[]
for i in range(B):
    cn=[(op[arch_normal[4*i+1]], arch_normal[4*i]),(op[arch_normal[4*i+3]], arch_normal[4*i+2])]
    cnr=[(op[arch_reduce[4*i+1]], arch_reduce[4*i]),(op[arch_reduce[4*i+3]], arch_reduce[4*i+2])]
    genotype["normal"].append(cn)
    genotype["reduce"].append(cnr)
genotype["normal_concat"]=range(2,2+B)
genotype["reduce_concat"]=range(2,2+B)

## Search

dataset = "Sport8" #or "MIT67"
batchsize = 16
datapath = "/data" #path to data
python train_search.py --dataset dataset --data datapath --batch_size batchsize --save test

## Augment

Same as DARTS
