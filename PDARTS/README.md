# [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation]

## Generate a Random Architecture 

Same as DARTS, with the restriction of 2 skip-connections maximum for a given architecture.

## Search

```
python train_search.py 
--save test 
--tmp_data_dir "/data" # path to data
--dataset cifar10 # choose between cifar10, cifar100, sport8, mit67 and flowers102
--layers 5 # 5 for cifar10 and cifar100, 8 for sport8, mit67 and flowers102
--add_layers 6 # 6 for cifar10 and cifar100, 0 for sport8, mit67 and flowers102
--add_layers 12 # 12 for cifar10 and cifar100, 0 for sport8, mit67 and flowers102
--dropout_rate 0.0 # 0.0 for cifar10, sport8, mit67 and flowers102, 0.1 for cifar100
--dropout_rate 0.4 # 0.4 for cifar10, sport8, mit67 and flowers102, 0.2 for cifar100
--dropout_rate 0.7 # 0.7 for cifar10, sport8, mit67 and flowers102, 0.3 for cifar100
```

## Augment

```
python train_cifar.py 
--save test 
--tmp_data_dir "/data" # path to data
--dataset cifar10 # choose between cifar10, cifar100, sport8, mit67 and flowers102
--layers 20 # 20 for cifar10 and cifar100, 8 for sport8, mit67 and flowers102
--arch genotype 
--auxiliary 
--cutout 
```

