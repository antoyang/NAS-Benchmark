# Neural Architecture Optimization

## Requirements
Pytorch >= 1.0.0

## Generate random architectures
sys.path.insert(0,'/cache/%s' % folder)
from utils import generate_arch
N = 6
B = 5
arch = generate_arch(1, B, num_ops=11)[0]
sys.path.pop(0)

## Search
dataset = "CIFAR10" #choose between CIFAR10, CIFAR100, Sport8 and MIT67
datapath = "/data" #path to data
layers = 3 #2 for Sport8 and MIT67
#child_eval_batch_size = 64 for Sport 8 and MIT67
python train_search.py --dataset dataset --child_batch_size 64 --child_layers layers --child_epochs 200 --child_eval_epochs 50 --controller_expand 8 --child_keep_prob 1.0 --child_drop_path_keep_prob 0.9  --child_sample_policy "params" --data datapath --output_dir test --dataset dataset

## Augment
python train_cifar.py --use_aux_head --keep_prob 0.6 --drop_path_keep_prob 0.8 --cutout_size 16 --l2_reg 3e-4 --arch arch --channels=36 --batch_size=128 --output_dir test --data datapath

