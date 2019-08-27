import re
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from darts.utils import create_exp_dir

class PerformanceView(object):
    def __init__(self, file_path):
        super(PerformanceView, self).__init__()
        self.valid_loss = []
        self.valid_acc = []
        self.train_loss = []
        self.train_acc = []
        self.file_path = file_path
        assert os.path.exists(file_path) == True
        self._read_log_file1()
        self.fig, self.axs = plt.subplots(nrows=1, ncols=2,
                                          sharex=False, figsize=(12,4), dpi=600)
        # self.fig.suptitle('train on cifar-10', fontsize=12, fontweight='bold', y=1.0)
        self.fig.subplots_adjust(left=0.2, wspace=0.2)
        self.fig.tight_layout()
        self.store_path = './performance_view/' + 'graph-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M%S'))
        create_exp_dir(self.store_path, scripts_to_save=None)

    def _read_log_file1(self):
        with open(self.file_path, 'r') as fp:
            for line in fp.readlines():
                if 'valid loss' in line:
                    line = line.strip().split(',')[1:]
                    for item in line:
                        if 'valid loss' in item:
                            self.valid_loss.append(float(item.split()[-1]))
                        else:
                            self.valid_acc.append(float(item.split()[3]))
                if 'train loss' in line:
                    line = line.strip().split(',')[1:]
                    for item in line:
                        if 'train loss' in item:
                            self.train_loss.append(float(item.split()[-1]))
                        else:
                            self.train_acc.append(float(item.split()[2]))


    def _read_log_file2(self):
        with open(self.file_path, 'r') as fp:
            for line in fp.readlines():
                if 'valid_acc' in line:
                    self.valid_acc.append(float(line.strip().split()[-1]))
                if 'train_acc' in line:
                    self.train_acc.append(float(line.strip().split()[-1]))
                if 'lr' in line:  # get learning rate
                    lr = float(line.strip().split()[-1])
                    self.lrs.append(lr)

    def draw_loss(self):

        self.axs[0].plot(self.valid_loss,'r', label='valid loss')
        self.axs[0].plot(self.train_loss, 'b', label='train loss')
        self.axs[0].set_xlabel('epoch')
        self.axs[0].set_ylabel('loss')
        self.axs[0].set_title('valid loss vs train loss each epoch')
        self.axs[0].legend()

    def draw_acc(self):
        self.axs[1].plot(self.valid_acc, 'r', label='valid acc')
        self.axs[1].plot(self.train_acc, 'b', label='train acc')
        self.axs[1].set_xlabel('epoch')
        self.axs[1].set_ylabel('accuracy')
        self.axs[1].set_title('valid accuracy vs train accuracy each epoch')
        self.axs[1].legend()

    def show(self):
        plt.savefig(self.store_path+'/v3_loss_acc_cifar100.pdf', bbox_inches = 'tight', dpi=600)
        plt.show()

if __name__ == '__main__':
   #pv = PerformanceView('../logs/eval/DARTS_MORE_NONE_V2/cifar10/eval-EXP-20181021-1427/log.txt')
   #pv = PerformanceView('../logs/eval/DARTS_MORE_NONE_V1/cifar100/eval-EXP-20181021-0200/log.txt')
   #pv = PerformanceView('../logs/eval/DARTS_MORE_NONE_V3/cifar10/eval-EXP-20181021-0206/log.txt')
   #pv = PerformanceView('../logs/eval/DARTS_MORE_NONE_V3/cifar100/eval-EXP-20181025-0243/log.txt')
   pv = PerformanceView('../logs/eval/DARTS_MORE_V3/tiny-imagenet\eval-EXP-20181122-1325/log.txt')
   pv.draw_loss()
   pv.draw_acc()
   pv.show()


