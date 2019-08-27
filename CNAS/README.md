## CNAS 

## Requirements

Python >= 3.6
torch >= 0.4.1
torchvision == 0.2.1
seaborn (optional)
pygraphviz (optional)

## Generate random architectures
sys.path.insert(0, '/cache/%s' % folder)
import darts.geno_types as geno_types
import darts.model_search as models
model = models.Network(3,16,10,20,None,num_inp_node = 2, num_meta_node = 6)
genotype = model.genotype()
sys.path.pop(0)

## Search

dataset = "cifar10" #choose between cifar10, cifar100, sport8 and mit67 
datapath = "/data" #path to data
python run/train_search.py --batch-size 64 --epochs 50 --num-meta-node 6 --cutout --data datapath --train-dataset dataset --save test 

## Augment

python run/train_cnn.py --epochs 600 --learning-rate 0.025 --batch-size 64 --drop-path-prob 0.25 --cutout --data datapath --train-dataset dataset --arch genotype --save test' 




