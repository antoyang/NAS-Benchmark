# DARTS: Differentiable Architecture Search

## Requirements

- python 3
- pytorch >= 0.4.1
- graphviz
    - First install using `apt install` and then `pip install`.
- numpy
- tensorboardX

## Generate a Random Architecture 

path = "/DARTS" #path to DARTS code
sys.path.insert(0, '/%s' % path)
from models.search_cnn import SearchCNNController
model = SearchCNNController(3, 16, 10, 20, None, n_nodes=4)
genotype = model.genotype()

## Search

datapath = "/data" #path to data
dataset = "CIFAR10" #choose between CIFAR10, CIFAR100, Sport8 and MIT67
python search.py --name test --layers 8 --epochs 50 --data_path datapath --dataset dataset

## Augment

layers = 20 #8 for Sport8 and MIT67
python augment.py --name test --layers layers --dataset dataset --datapath datapath --genotype genotype 
