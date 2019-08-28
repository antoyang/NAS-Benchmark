# DARTS: Differentiable Architecture Search

## Generate a Random Architecture 

```
from models.search_cnn import SearchCNNController
model = SearchCNNController(3, 16, 10, 20, None, n_nodes=4)
genotype = model.genotype()
```

## Search

```
python search.py 
--name test 
--data_path /data # path to data 
--dataset CIFAR10 # choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102
```

## Augment

```
python augment.py 
--name test 
--layers 20 # 20 for CIFAR10 and CIFAR100, 8 for Sport8, MIT67 and flowers102
--dataset CIFAR10 # choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102 
--datapath /data # path to data 
--genotype genotype 
```
