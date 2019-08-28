# Augmentation : repository used to evaluate different augmentation protocols

## Usage :
With autoaugment:

```
python augment.py --name test --dataset CIFAR10 --data_path %s --epochs %d --init_channels %d --genotype "%s" --cutout_length %d --drop_path_prob %d --aux_weight %d --seed %d --autoaugment
     %   (datapath, epochs, channels, genotype, cutout_length, drop_path_prob, aux_weight, seed)
```
