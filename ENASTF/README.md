# Efficient Neural Architecture Search via Parameter Sharing

Original code used for CIFAR-10 and CIFAR-100 experiments

## Generate a random architecture

import numpy.random as rd
ops = rd.randint(0, 5, 20)
links = rd.choice([0, 1], size=20, replace=True)
arch_conv = "%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i " % (
links[0], ops[0], links[1], ops[1], links[0], ops[2], links[1], ops[3], links[0], ops[4], links[1], ops[5],
links[0], ops[6], links[1], ops[7], links[0], ops[16], links[1], ops[17])
arch_red = "%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i" % (
links[0], ops[8], links[1], ops[9], links[0], ops[10], links[1], ops[11], links[0], ops[12], links[1], ops[13],
links[0], ops[14], links[1], ops[15], links[0], ops[18], links[1], ops[19])
arch = arch_conv + arch_red

## Search

dataset = "CIFAR10" #or "CIFAR100"
datapath = "/data/CIFAR10" #path to data
python src/cifar10/main.py --data_format="NCHW" --search_for="micro" --reset_output_dir --data_path=datapath --dataset dataset --output_dir test --batch_size=160 --num_epochs=150 --log_every=50 --eval_every_epochs=1 --child_use_aux_heads --child_num_layers=6 --child_out_filters=20 --child_l2_reg=1e-4 --child_num_branches=5 --child_num_cells=5 --child_keep_prob=0.90 --child_drop_path_keep_prob=0.60 --child_lr_cosine --child_lr_max=0.05 --child_lr_min=0.0005 --child_lr_T_0=10 --child_lr_T_mul=2 --controller_training --controller_search_whole_channels --controller_entropy_weight=0.0001 --controller_train_every=1 --controller_sync_replicas --controller_num_aggregate=10 --controller_train_steps=30 --controller_lr=0.0035 --controller_tanh_constant=1.10 --controller_op_tanh_reduce=2.5

## Augment

python src/cifar10/main.py --data_format="NCHW" --search_for="micro" --reset_output_dir --data_path=datapath --output_dir test --batch_size=144 --num_epochs=630 --log_every=50 --eval_every_epochs=1 --child_fixed_arc=arch --child_use_aux_heads --child_num_layers=15 --child_out_filters=36 --child_num_branches=5 --child_num_cells=5 --child_keep_prob=0.80 --child_drop_path_keep_prob=0.60 --child_l2_reg=2e-4 --child_lr_cosine --child_lr_max=0.05 --child_lr_min=0.0001 --child_lr_T_0=10 --child_lr_T_mul=2 --nocontroller_training --controller_search_whole_channels --controller_entropy_weight=0.0001 --controller_train_every=1 --controller_sync_replicas --controller_num_aggregate=10 --controller_train_steps=50 --controller_lr=0.001 --controller_tanh_constant=1.50 --controller_op_tanh_reduce=2.5 --dataset dataset
