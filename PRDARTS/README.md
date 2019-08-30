# Prune and Replace NAS
by Kevin Alexander Laube and Andreas Zell, [on arxiv](https://arxiv.org/abs/1906.07528)

**This code is based on the implementations of
[DARTS](https://github.com/quark0/darts) and
[P-DARTS](https://github.com/chenxin061/pdarts).**

## Idea:

Iteratively pruning and replacing bad candidates from the operation pool enables us
to efficiently search through vast operation spaces in reasonable time.
We use DARTS to rank the currently available candidates, prune the worst ones,
and generate new ones by applying network morphisms to those that are left.

## Run:

Sample scripts that match our experiments in the paper are provided
in the ./scripts folder.

#### Search

The progress and discovered cells of our DL2 search, these stats are plotted to tensorboard.


<p align="center">
    <img src="images/DL2_epoch_stats-1.png" alt="Search" width="80%"/>
</p>

<p align="center">
    <img src="images/DL2_normal-1.png" alt="DL2normal" width=49%/>
    <img src="images/DL2_reduction-1.png" alt="DL2reduction" width=49%/>
</p>



#### Retraining

Test error (%) on CIFAR-10 and CIFAR-100 in comparison.
The focus of this research is the candidate operation space,
thus the only major change compared to the DARTS baseline is
the progressive pruning and replacement schedule.

Method                                                  | #params   | #ops      | GPU days  | CIFAR-10  | CIFAR-100
-------                                                 |---------  |------     |---------- |-----------|-----------
[NASNet-A](https://arxiv.org/abs/1707.07012)            | 3.3M      | 13        | 1800      | 2.65
[AmoebaNet-B](https://arxiv.org/abs/1802.01548)         | 2.8M      | 19        | 3150      | 2.55
[ENAS](https://arxiv.org/abs/1802.03268)                | 4.6M      | 5         | 0.5       | 2.89
[DARTS (1st order)](https://arxiv.org/abs/1806.09055)   | 2.9M      | 8         | 1.5       | 2.94
[DARTS (2nd order)](https://arxiv.org/abs/1806.09055)   | 3.4M      | 8         | 4         | 2.83
[P-DARTS C10](https://arxiv.org/abs/1904.12760)         | 3.4M      | 8         | 0.3       | 2.5       | 16.55
[P-DARTS C100](https://arxiv.org/abs/1904.12760)        | 3.6M      | 8         | 0.3       | 2.62      | 15.92
[sharpDARTS](https://arxiv.org/abs/1903.09900)          | 3.6M      |           | 0.8       | 2.45
[SNAS moderate](https://arxiv.org/abs/1812.09926)       | 2.8M      | 8         | 1.5       | 2.85
[NASP](https://arxiv.org/abs/1905.13577)                | 3.3M      | 7         | 0.2       | 2.8
[NASP (more ops)](https://arxiv.org/abs/1905.13577)     | 7.4M      | 12        | 0.3       | 2.5
PR-DARTS DL1                                            | 3.2M      | 15/15     | 0.82      | 2.74      | 17.37
PR-DARTS DL2                                            | 4.0M      | 15/15     | 0.82      | 2.51      | 15.53
PR-DARTS DR                                             | 4.2M      | 26/39     | 0.88      | 2.55      | 16.69
PR-DARTS UR                                             | 5.4M      | 45/83     | 1.10      | 3.79      | 


## Citation
If you use any part of this code in your research, please cite our [paper](https://arxiv.org/abs/1906.07528):
```
@article{laube2019prnas,
  title={Prune and Replace NAS},
  author={Kevin A. Laube and Andreas Zell},
  journal={arXiv preprint https://arxiv.org/abs/1906.07528},
  year={2019}
}
```
