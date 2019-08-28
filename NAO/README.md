# Neural Architecture Optimization

## Generate random architectures

```
from utils import generate_arch
arch = generate_arch(1, 5, num_ops=11)[0]
```

## Search

```
python train_search.py 
--dataset CIFAR10 #choose between CIFAR10, CIFAR100, Sport8, MIT67 and flowers102
--child_batch_size 64 
--child_eval_batch_size 500 # 500 for CIFAR10 and CIFAR100, 128 for Sport8, MIT67 and flowers102
--child_layers 3 # 3 for CIFAR10 and CIFAR100, 2 for Sport8, MIT67 and flowers102
--child_epochs 200 
--child_eval_epochs 50 
--controller_expand 8 
--child_keep_prob 1.0 
--child_drop_path_keep_prob 0.9  
--child_sample_policy "params" 
--data /data #path to data
--output_dir test 
```

## Augment

For CIFAR datasets:

```
python train_cifar.py 
--use_aux_head 
--keep_prob 0.6 
--drop_path_keep_prob 0.8 
--cutout_size 16 
--l2_reg 3e-4 
--arch arch 
--channels=36 
--batch_size=128 
--output_dir test 
--data /data # path to data
--dataset CIFAR10 # choose between CIFAR10 and CIFAR100
```

For other datasets :

```
python train_imagenet.py 
--use_aux_head 
--keep_prob 0.6 
--drop_path_keep_prob 0.8 
--cutout_size 16 
--l2_reg 3e-4 
--arch arch 
--channels 36 
--batch_size 96 
--layers 2 
--epochs 600 
--output_dir test
--data /data # path to data
--dataset sport8 # choose between sport8, mit67 and flowers102
```


