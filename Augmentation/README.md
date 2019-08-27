# Augmentation : repository used to evaluate different augmentation protocols

## Requirements

- python 3
- pytorch >= 0.4.1
- graphviz
    - First install using `apt install` and then `pip install`.
- numpy
- tensorboardX

## Usage :
Avec AutoAugment:
python augment.py --name test --dataset CIFAR10 --data_path %s --epochs %d --init_channels %d --genotype "%s" --cutout_length %d --drop_path_prob %d --aux_weight %d --seed %d --autoaugment
     %   (datapath, epochs, channels, genotype, cutout_length, drop_path_prob, aux_weight, seed)
