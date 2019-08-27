# [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation]

## Requirements 

Same as DARTS

## Generate a Random Architecture 

Same as DARTS, with the restriction of 2 skip-connections maximum for a given architecture.

## Search

datapath = "/data" #path to data
dataset = "cifar10" #choose between cifar10, cifar100, sport8 and mit67
#For cifar10:
python train_search.py --save test --tmp_data_dir datapath --dataset dataset --layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7
#For cifar100:
python train_search.py --save test --tmp_data_dir datapath --dataset dataset --layers 5 --add_layers 6 --add_layers 12 --dropout_rate 0.1 --dropout_rate 0.2 --dropout_rate 0.3 for CIFAR100
#For mit67 and sport8 datasets:
python train_search.py --save test --tmp_data_dir datapath --dataset dataset --layers 8 --add_layers 0 --add_layers 0 --dropout_rate 0.0 --dropout_rate 0.4 --dropout_rate 0.7

## Augment

layers = 20 #8 for Sport8 and MIT67
python train_cifar.py --tmp_data_dir datapath --dataset dataset --arch genotype --layers layers --auxiliary --cutout --save test 

