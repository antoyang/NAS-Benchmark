# Augmentation : repository used to evaluate different augmentation protocols

## Generate a Random Architecture 

```
from models.search_cnn import SearchCNNController
model = SearchCNNController(3, 16, 10, 20, None, n_nodes=4)
genotype = model.genotype()
```

## Augmentation :

```
python augment.py 
--name test 
--dataset CIFAR10 
--data_path /data # path to data 
--epochs 600 # or 1500 depending on experiments
--init_channels 36 # or 50 depending on experiments
--genotype genotype
--cutout_length 16 # or 0 depending on experiments
--drop_path_prob 0.2 # or 0 depending on experiments
--aux_weight 0.4 # or 0 depending on experiments
--seed 2 # or another one depending on experiments
--autoaugment # remove not to use it
```
