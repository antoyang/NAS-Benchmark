import os
import sys
import time
import logging
import torch
import argparse
import numpy as np
import torch.utils
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from adabound import AdaBound
import random

sys.path.append('/cache/CNAS') #Path to your code repository
import darts.utils as dutils
import darts.datasets as dartsdset
import darts.geno_types as geno_types
from darts.model import EvalNetwork, CrossEntropyLabelSmooth

parser = argparse.ArgumentParser("cifar or imagenet")
parser.add_argument('--train-dataset', type=str, default='cifar10', help='training data')
parser.add_argument('--data', type=str, default='/train_tiny_data/train_data/cifar10', help='location of the data corpus')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-classes', type=int, default=10, help='num of classes')
parser.add_argument('--learning-rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report-freq', type=float, default=50, help='report frequency')
parser.add_argument('--multi-gpus', action='store_true', default=False, help='use multi gpus')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init-channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model-path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary-weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--opt', type=str, default='sgd', help='use sgd')
parser.add_argument('--cutout-length', type=int, default=16, help='cutout length')
parser.add_argument('--drop-path-prob', type=float, default=0.05, help='drop path probability')
parser.add_argument('--no-dropout', action='store_true', default=False, help='use dropout')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay-period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

class TrainNetwork(object):
    """The main train network"""

    def __init__(self, args):
        super(TrainNetwork, self).__init__()
        self.args = args
        self.dur_time = 0
        self.logger = self._init_log()

        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)

        self._init_hyperparam()
        self._init_random_and_device()
        self._init_model()

    def _init_hyperparam(self):
        if 'cifar10' == self.args.train_dataset:
            # cifar10:  6000 images per class, 10 classes, 50000 training images and 10000 test images
            # cifar100: 600 images per class, 100 classes, 500 training images and 100 testing images per class
            self.args.num_classes = 10
            self.args.layers = 20
            self.args.data = self.args.data
        elif 'cifar100' == self.args.train_dataset:
            # cifar10:  6000 images per class, 10 classes, 50000 training images and 10000 test images
            # cifar100: 600 images per class, 100 classes, 500 training images and 100 testing images per class
            self.args.num_classes = 100
            self.args.layers = 20
            self.args.data = self.args.data
        elif 'imagenet' == self.args.train_dataset:
            self.args.data = '/train_data/imagenet'
            self.args.num_classes = 1000
            self.args.weight_decay = 3e-5
            self.args.report_freq = 100
            self.args.init_channels = 50
            self.args.drop_path_prob = 0
        elif 'tiny-imagenet' == self.args.train_dataset:
            self.args.data = '/train_tiny_data/train_data/tiny-imagenet'
            self.args.num_classes = 200
        elif 'food101' == self.args.train_dataset:
            self.args.data = '/train_tiny_data/train_data/food-101'
            self.args.num_classes = 101
            self.args.init_channels = 48
        elif self.args.train_dataset == "mit67":
            self.args.num_classes = 67
        elif self.args.train_dataset == "sport8":
            self.args.num_classes = 8
        elif self.args.train_dataset == "flowers102":
            self.args.num_classes = 102

    def _init_log(self):
        self.args.save = self.args.save
        dutils.create_exp_dir(self.args.save, scripts_to_save=None)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger('Architecture Training')
        logger.addHandler(fh)
        return logger

    def _init_random_and_device(self):
        # Set random seed and cuda device
        np.random.seed(self.args.seed)
        cudnn.benchmark = True
        torch.manual_seed(self.args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.args.seed)
        max_free_gpu_id, gpus_info = dutils.get_gpus_memory_info()
        self.device_id = max_free_gpu_id
        self.gpus_info = gpus_info
        self.device = torch.device('cuda:{}'.format(0 if self.args.multi_gpus else self.device_id))

    def _init_model(self):

        self.train_queue, self.valid_queue = self._load_dataset_queue()

        def _init_scheduler():
            if 'cifar' or 'mit' or 'sport' or 'flowers' in self.args.train_dataset:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(self.args.epochs))
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.decay_period,
                                                            gamma=self.args.gamma)
            return scheduler

        genotype = eval('geno_types.%s' % self.args.arch)
        reduce_level = (0 if 'cifar10' in self.args.train_dataset else 3)
        model = EvalNetwork(self.args.init_channels, self.args.num_classes, 0,
                            self.args.layers, self.args.auxiliary, genotype, reduce_level)

        # Try move model to multi gpus
        if torch.cuda.device_count() > 1 and self.args.multi_gpus:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)

        self.logger.info('param size = %fM', dutils.calc_parameters_count(model))

        criterion = nn.CrossEntropyLoss()
        if self.args.num_classes >= 103:
            criterion = CrossEntropyLabelSmooth(self.args.num_classes, self.args.label_smooth)
        self.criterion = criterion.to(self.device)

        if self.args.opt == 'adam':
            self.optimizer = torch.optim.Adamax(
                model.parameters(),
                self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.opt == 'adabound':
            self.optimizer = AdaBound(model.parameters(),
            self.args.learning_rate,
            weight_decay=self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )

        self.best_acc_top1 = 0
        # optionally resume from a checkpoint
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint {}".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.dur_time = checkpoint['dur_time']
                self.args.start_epoch = checkpoint['epoch']
                self.best_acc_top1 = checkpoint['best_acc_top1']
                self.args.drop_path_prob = checkpoint['drop_path_prob']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        self.scheduler = _init_scheduler()
        # reload the scheduler if possible
        if self.args.resume and os.path.isfile(self.args.resume):
            checkpoint = torch.load(self.args.resume)
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _load_dataset_queue(self):
        if 'cifar' in self.args.train_dataset:
            train_transform, valid_transform = dutils.data_transforms_cifar(self.args)
            if 'cifar10' == self.args.train_dataset:
                train_data = dset.CIFAR10(root=self.args.data, train=True, download=True, transform=train_transform)
                valid_data = dset.CIFAR10(root=self.args.data, train=False, download=True, transform=valid_transform)
            else:
                train_data = dset.CIFAR100(root=self.args.data, train=True, download=True, transform=train_transform)
                valid_data = dset.CIFAR100(root=self.args.data, train=False, download=True, transform=valid_transform)

            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size = self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
            )
            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size = self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
            )
        elif 'tiny-imagenet' == self.args.train_dataset:
            train_transform, valid_transform = dutils.data_transforms_tiny_imagenet()
            train_data = dartsdset.TinyImageNet200(self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dartsdset.TinyImageNet200(self.args.data, train=False, download=True, transform=valid_transform)
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
            )
            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
            )
        elif 'imagenet' == self.args.train_dataset:
            traindir = os.path.join(self.args.data, 'train')
            validdir = os.path.join(self.args.data, 'val')
            train_transform, valid_transform = dutils.data_transforms_imagenet()
            train_data = dset.ImageFolder(
                traindir,train_transform)
            valid_data = dset.ImageFolder(
                validdir,valid_transform)

            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        elif 'food101' == self.args.train_dataset:
            traindir = os.path.join(self.args.data, 'train')
            validdir = os.path.join(self.args.data, 'val')
            train_transform, valid_transform = dutils.data_transforms_food101()
            train_data = dset.ImageFolder(
                traindir,train_transform)
            valid_data = dset.ImageFolder(
                validdir,valid_transform)

            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        elif 'mit67' == self.args.train_dataset:
            dset_cls = dset.ImageFolder
            n_classes = 67
            data_path = '%s/MIT67/train' % self.args.data  # 'data/MIT67/train'
            val_path = '%s/MIT67/test' % self.args.data  # 'data/MIT67/val'
            train_transform, valid_transform = dutils.data_transforms_large(self.args.train_dataset,self.args.cutout_length)
            train_data = dset_cls(root=data_path, transform=train_transform)
            valid_data = dset_cls(root=val_path, transform=valid_transform)
            shape = train_data[0][0].unsqueeze(0).shape
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            train_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       num_workers=4,
                                                       pin_memory=True)
            valid_queue = torch.utils.data.DataLoader(valid_data,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       pin_memory=True)

        elif 'sport8' == self.args.train_dataset:
            dset_cls = dset.ImageFolder
            n_classes = 8
            data_path = '%s/Sport8/train' % self.args.data  
            val_path = '%s/Sport8/test' % self.args.data  
            train_transform, valid_transform = dutils.data_transforms_large(self.args.train_dataset,
                                                                            self.args.cutout_length)
            train_data = dset_cls(root=data_path, transform=train_transform)
            valid_data = dset_cls(root=val_path, transform=valid_transform)
            shape = train_data[0][0].unsqueeze(0).shape
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            train_queue = torch.utils.data.DataLoader(train_data,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      pin_memory=True)
            valid_queue = torch.utils.data.DataLoader(valid_data,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)
        elif 'flowers102' == self.args.train_dataset:
            dset_cls = dset.ImageFolder
            n_classes = 102
            data_path = '%s/flowers102/train' % self.args.data
            val_path = '%s/flowers102/test' % self.args.data
            train_transform, valid_transform = dutils.data_transforms_large(self.args.train_dataset,
                                                                            self.args.cutout_length)
            train_data = dset_cls(root=data_path, transform=train_transform)
            valid_data = dset_cls(root=val_path, transform=valid_transform)
            shape = train_data[0][0].unsqueeze(0).shape
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            train_queue = torch.utils.data.DataLoader(train_data,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      num_workers=4,
                                                      pin_memory=True)
            valid_queue = torch.utils.data.DataLoader(valid_data,
                                                      batch_size=self.args.batch_size,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      pin_memory=True)
        return train_queue, valid_queue

    def run(self):
        self.logger.info('args = %s', self.args)
        run_start = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.scheduler.step()
            self.logger.info('epoch % d / %d  lr %e', epoch, self.args.epochs, self.scheduler.get_lr()[0])

            if self.args.no_dropout:
                self.model._drop_path_prob = 0
            else:
                self.model._drop_path_prob = self.args.drop_path_prob * epoch / self.args.epochs
                self.logger.info('drop_path_prob %e', self.model._drop_path_prob)

            train_acc, train_obj = self.train()
            self.logger.info('train loss %e, train acc %f', train_obj, train_acc)

            valid_acc_top1, valid_acc_top5, valid_obj = self.infer()
            self.logger.info('valid loss %e, top1 valid acc %f top5 valid acc %f',
                        valid_obj, valid_acc_top1, valid_acc_top5)
            self.logger.info('best valid acc %f', self.best_acc_top1)

            is_best = False
            if valid_acc_top1 > self.best_acc_top1:
                self.best_acc_top1 = valid_acc_top1
                is_best = True

            dutils.save_checkpoint({
                'epoch': epoch+1,
                'dur_time': self.dur_time + time.time() - run_start,
                'state_dict': self.model.state_dict(),
                'drop_path_prob': self.args.drop_path_prob,
                'best_acc_top1': self.best_acc_top1,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }, is_best, self.args.save)
        self.logger.info('train epoches %d, best_acc_top1 %f, dur_time %s',
                         self.args.epochs, self.best_acc_top1, dutils.calc_time(self.dur_time + time.time() - run_start))

    def train(self):
        objs = dutils.AverageMeter()
        top1 = dutils.AverageMeter()
        top5 = dutils.AverageMeter()

        self.model.train()

        for step, (input, target) in enumerate(self.train_queue):

            input = input.cuda(self.device, non_blocking=True)
            target = target.cuda(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits, logits_aux = self.model(input)
            loss = self.criterion(logits, target)
            if self.args.auxiliary:
                loss_aux = self.criterion(logits_aux, target)
                loss += self.args.auxiliary_weight*loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            prec1, prec5 = dutils.accuracy(logits, target, topk=(1,5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                self.logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def infer(self):
        objs = dutils.AverageMeter()
        top1 = dutils.AverageMeter()
        top5 = dutils.AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_queue):
                input = input.cuda(self.device, non_blocking=True)
                target = target.cuda(self.device, non_blocking=True)

                logits, _ = self.model(input)
                loss = self.criterion(logits, target)

                prec1, prec5 = dutils.accuracy(logits, target, topk=(1,5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % args.report_freq == 0:
                    self.logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    args = parser.parse_args()
    train_network = TrainNetwork(args)
    train_network.run()
