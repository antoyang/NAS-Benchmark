# NAS-Benchmark

This repository includes the code used to evaluate NAS methods on 5 different datasets, as well as the code used to augment architectures with different protocols, as mentioned in our ICLR2020 paper (https://arxiv.org/abs/1912.12522). Scripts exemples are provided in each folder.

## News
ICLR2020 will be hosted virtually and we will be available for discussion and questions during our Poster Sessions, Wednesday 6 12:00 to 14:00 and 17:00 to 19:00 GMT (https://iclr.cc/virtual/poster_HygrdpVKvr.html).

A 5 min video (with associated slides) presenting the paper will also be made public for the conference.

## Plots
All code used to generate the plots of the paper can be found in the "Plots" folder.

## Randomly Sampled Architectures
You can find all sampled architectures and corresponding training logs in Plots\data\modified_search_space.

## Data

In the data folder, you will find the data splits for Sport-8, MIT-67 and Flowers-102 in .csv files.

You can download these datasets on the following web sites :

Sport-8 : http://vision.stanford.edu/lijiali/event_dataset/

MIT-67 : http://web.mit.edu/torralba/www/indoor.html

Flowers-102 : http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

The data path has to be set the following way : dataset/train/classes/images for the training set, dataset/test/classes/images for the test set.

We used the following repositories :

## DARTS
Paper : Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). 

Unofficial updated implementation : https://github.com/khanrc/pt.darts

## P-DARTS
Paper : Xin Chen, Lingxi Xie, Jun Wu, Qi Tian. "Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation." ICCV, 2019.

Official implementation : https://github.com/chenxin061/pdarts

## CNAS
Paper : Weng, Yu, et al. "Automatic Convolutional Neural Architecture Search for Image Classification Under Different Scenes." IEEE Access 7 (2019): 38495-38506.

Official implementation : https://github.com/tianbaochou/CNAS

## StacNAS
Paper : Guilin Li et al. "StacNAS: Towards Stable and Consistent Differentiable Neural Architecture Search." 	arXiv preprint  arXiv:1909.11926 (2019).

Implementation : provided by the authors

## ENAS
Paper : Pham, Hieu, et al. "Efficient neural architecture search via parameter sharing." arXiv preprint arXiv:1802.03268 (2018).

Official Tensorflow implementation : https://github.com/melodyguan/enas

Unofficial Pytorch implementation : https://github.com/MengTianjian/enas-pytorch

## MANAS 
Paper : Maria Carlucci, Fabio, et al. "MANAS: Multi-Agent Neural Architecture Search." arXiv preprint arXiv:1909.01051 (2019).

Implementation : provided by the authors. 

## NSGA-NET
Paper : Lu, Zhichao, et al. "NSGA-NET: a multi-objective genetic algorithm for neural architecture search." arXiv preprint arXiv:1810.03522 (2018).

Official implementation : https://github.com/ianwhale/nsga-net

## NAO 
Paper : Luo, Renqian, et al. "Neural architecture optimization." Advances in neural information processing systems. 2018.

Official Pytorch implementation : https://github.com/renqianluo/NAO_pytorch


For the two following methods, we have not yet performed consistent experiments (therefore the methods are not included in the paper). Nonetheless, we provide runnable code that could provide relevant insights (similar to those provided in the paper on the other methods) on these methods.

## PC-DARTS 
Paper : Xu, Yuhui, et al. "PC-DARTS: Partial Channel Connections for Memory-Efficient Differentiable Architecture Search." arXiv preprint arXiv:1907.05737 (2019).

Official implementation : https://github.com/yuhuixu1993/PC-DARTS

## PRDARTS
Paper : Laube, Kevin Alexander, and Andreas Zell. "Prune and Replace NAS." arXiv preprint arXiv:1906.07528 (2019).

Official implementation : https://github.com/cogsys-tuebingen/prdarts

## AutoAugment
Paper : Cubuk, Ekin D., et al. "Autoaugment: Learning augmentation policies from data." arXiv preprint arXiv:1805.09501 (2018).

Unofficial Pytorch implementation : https://github.com/DeepVoltaire/AutoAugment

## Citation

If you found this work useful, consider citing for now:

```
@misc{yang2019nas,
    title={NAS evaluation is frustratingly hard},
    author={Antoine Yang and Pedro M. Esperança and Fabio M. Carlucci},
    year={2019},
    eprint={1912.12522},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
