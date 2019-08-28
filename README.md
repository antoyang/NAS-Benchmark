# NAS-Benchmark

This repository includes the codes used to evaluate NAS methods on 5 different datasets, as well as the code used to augment architectures with different protocols, as mentioned in our paper. Scripts exemples are provided in each folder.

# Data

In the data folder, you will find the data splits for Sport-8, MIT-67 and Flowers-102 in .csv files.

You can download these datasets on the following sites :

Sport-8 : http://vision.stanford.edu/lijiali/event_dataset/

MIT-67 : http://web.mit.edu/torralba/www/indoor.html

Flowers-102 : http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

The data path has to be set the following way : dataset/train/classes/images for the training set, dataset/test/classes/images for the test set.

We used the following repositories :

# DARTS
Paper : Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). 

Unofficial updated implementation : https://github.com/khanrc/pt.darts

# P-DARTS
Paper : Xin Chen, Lingxi Xie, Jun Wu, Qi Tian, Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation, ICCV, 2019.
Official implementation : https://github.com/chenxin061/pdarts

# CNAS
Paper : Weng, Yu, et al. "Automatic Convolutional Neural Architecture Search for Image Classification Under Different Scenes." IEEE Access 7 (2019): 38495-38506.

Official implementation : https://github.com/tianbaochou/CNAS

# iDARTS
Paper : To be released

Implementation : provided by the authors

# ENAS
Paper : Pham, Hieu, et al. "Efficient neural architecture search via parameter sharing." arXiv preprint arXiv:1802.03268 (2018).

Official Tensorflow implementation : https://github.com/melodyguan/enas

Unofficial Pytorch implementation : https://github.com/MengTianjian/enas-pytorch

# MANAS 
Paper : To be released

Implementation : provided by the authors

# NSGA-NET
Paper : Lu, Zhichao, et al. "NSGA-NET: a multi-objective genetic algorithm for neural architecture search." arXiv preprint arXiv:1810.03522 (2018).

Official implementation : https://github.com/ianwhale/nsga-net

# NAO 
Paper : Luo, Renqian, et al. "Neural architecture optimization." Advances in neural information processing systems. 2018.

Official Pytorch implementation : https://github.com/renqianluo/NAO_pytorch

# AutoAugment
Paper : Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018).

Unofficial Pytorch implementation : https://github.com/DeepVoltaire/AutoAugment

