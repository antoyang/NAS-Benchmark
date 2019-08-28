# Automatic Convolutional Neural Architecture Search for Image Classification Under Different Scenes

## Generate random architectures

```
import darts.geno_types as geno_types
import darts.model_search as models
model = models.Network(3, 16, 10, 20, None, num_inp_node = 2, num_meta_node = 6)
genotype = model.genotype()
```

## Search

```
python run/train_search.py 
--batch-size 64 
--epochs 50 
--num-meta-node 6 
--cutout 
--data /data #path to data
--train-dataset cifar10 # choose between cifar10, cifar100, sport8, mit67 and flowers102
--save test 
```

## Augment

```
python run/train_cnn.py 
--epochs 600 
--learning-rate 0.025 
--batch-size 64 
--drop-path-prob 0.25 
--cutout 
--data /data # path to data
--train-dataset cifar10 # choose between cifar10, cifar100, sport8, mit67 and flowers102
--arch genotype 
--save test
```



