# NSGA-Net

## Requirements

Python >= 3.6.8, PyTorch >= 1.0.1.post2, torchvision >= 0.2.2, pymoo >= 0.3.1.dev0

## Generate random architectures
sys.path.insert(0, '/cache/%s' % folder)
from search.micro_encoding import decode
import numpy.random as rd

ops=rd.randint(0,8,16)
links0=rd.choice([0,1], size=2, replace=False)
links1=rd.choice([0,1,2], size=2, replace=False)
links2=rd.choice([0,1,2,3], size=2, replace=False)
links3=rd.choice([0,1,2,3,4], size=2, replace=False)
links0r=rd.choice([0,1], size=2, replace=False)
links1r=rd.choice([0,1,2], size=2, replace=False)
links2r=rd.choice([0,1,2,3], size=2, replace=False)
links3r=rd.choice([0,1,2,3,4], size=2, replace=False)
genome = [[[[ops[0], links0[0]], [ops[1], links0[1]]], [[ops[2], links1[0]], [ops[3], links1[1]]], [[ops[4], links2[0]], [ops[5], links2[1]]], [[ops[6], links3[0]], [ops[7], links3[1]]]],
         [[[ops[8], links0r[0]], [ops[9], links0r[1]]], [[ops[10], links1r[0]], [ops[11], links1r[1]]], [[ops[12], links2r[0]], [ops[13], links2r[1]]], [[ops[14], links3r[0]], [ops[15], links3r[1]]]]]
genotype = decode(genome)
sys.path.pop(0)

## Search
dataset = "CIFAR10" #choose between CIFAR10, CIFAR100, Sport8 and MIT67
datapath = "/data" #path to data
python search/evolution_search.py --init_channels 16 --layers 8 --epochs 20 --n_offspring 20 --n_gens 30 --search_space micro --save test --data_path datapath --dataset dataset

## Augment
layers = 20 #8 for Sport8 and MIT67
python validation/train.py --dataset dataset --net_type micro --layers layers --init_channels 34 --filter_increment 4  --cutout --auxiliary --batch_size 96 --droprate 0.2 --SE --epochs 600 --genotype genotype --data datapath --save test --path test

