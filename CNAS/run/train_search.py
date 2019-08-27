import os
import sys
import time
import glob
import torch
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import numpy as np
sys.path.append('/cache/CNAS') #path to your code repository
import darts.utils as dutils
from darts.model_search import Network
from darts.architecture import Architecture
import darts.datasets as dartsdset
import random

parser = argparse.ArgumentParser('Architecture Searching')
parser.add_argument('--train-dataset', type=str, default='cifar10',
                    help='training cifar10 or cifar100')
parser.add_argument('--arch', type=str, default='DARTS', help='the search arch name')
parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--num-meta-node', type=int, default=4, help='number of meta node in a cell')
parser.add_argument('--learning-rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning-rate-min', type=float, default=0.003, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report-freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--init-channels', type=int, default=16, help='num of init channels')
parser.add_argument('--image-channels', type=int, default=3, help='num of image channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--model-path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout-length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train-portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--sec-approx', action='store_true', default=False, help='use 2 order approximate validation loss')
parser.add_argument('--use-sparse', action='store_true', default=False, help='use sparse framework')
parser.add_argument('--arch-learning-rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch-weight-decay', type=float, default=1e-3, help='weight decay for arch encoding')


class SearchNetwork(object):
    """Search network in cifar10,cifar100 and tiny-imagenet"""

    def __init__(self, args):
        self.args = args
        self.logger = self._init_log()
        self.dur_time = 0

        self._init_hyperparam()
        self._init_device()
        self._load_data_queue()
        self._init_model()
        self._check_resume()

    def _init_log(self):
        self.args.save = self.args.save
        dutils.create_exp_dir(self.args.save, scripts_to_save=None)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger('Architecture Search')
        logger.addHandler(fh)
        return logger

    def _init_hyperparam(self):
        if self.args.train_dataset == 'cifar100':
            self.args.data = self.args.data
            self.args.epochs = 50
            self.args.init_channels = 24
            self.args.train_classes = 100
        elif self.args.train_dataset == 'cifar10':
            self.args.train_classes = 10
            self.args.data = self.args.data
        elif self.args.train_dataset == 'food101':
            self.args.train_classes = 101
            self.args.data = '/train_tiny_data/train_data/food-101'
        elif self.args.train_dataset == 'tiny-imagenet':
            self.args.train_classes = 200
            self.args.data = '/train_tiny_data/train_data/tiny-imagenet'
        elif self.args.train_dataset == "mit67":
            self.args.train_classes = 67
        elif self.args.train_dataset == "sport8":
            self.args.train_classes = 8
        elif self.args.train_dataset == "flowers102":
            self.args.train_classes = 102

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)
        self.current_gpu, _ = dutils.get_gpus_memory_info()
        np.random.seed(self.args.seed)
        torch.cuda.set_device(self.current_gpu)
        cudnn.benchmark = True
        torch.manual_seed(self.args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.args.seed)
        self.logger.info('gpu device = %d', self.current_gpu)

    def _check_resume(self):
        # optionally resume from a checkpoint for model
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.args.start_epoch = checkpoint['epoch']
                self.dur_time = checkpoint['dur_time']
                self.args.use_sparse = checkpoint['use_sparse']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model.alphas_normal = checkpoint['alphas_normal']
                self.model.alphas_reduce = checkpoint['alphas_reduce']
                self.model._arch_parameters = checkpoint['arch_parameter']
                self.architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                print('=> loaded checkpoint \'{}\'(epoch {})'.format(self.args.resume, self.args.start_epoch))
            else:
                print('=> no checkpoint found at \'{}\''.format(self.args.resume))

    def _load_data_queue(self):
        if 'cifar' in self.args.train_dataset:
            train_transform, valid_transform = dutils.data_transforms_cifar(self.args)
            train_data = (dset.CIFAR10(root=self.args.data, train=True, download=True,
                                       transform=train_transform) if 'cifar10' == self.args.train_dataset
                          else dset.CIFAR100(root=self.args.data, train=True, download=True, transform=train_transform))

            num_train = len(train_data)
            indices = list(range(num_train))
            split = int(np.floor(self.args.train_portion * num_train))

            self.train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True, num_workers=0
            )

            self.valid_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                pin_memory=True, num_workers=0
            )
        elif 'mit67' == self.args.train_dataset:
            dset_cls = dset.ImageFolder
            n_classes = 67
            data_path = '%s/MIT67/train' % self.args.data 
            val_path = '%s/MIT67/test' % self.args.data  
            train_transform, valid_transform = dutils.data_transforms_large(self.args.train_dataset,self.args.cutout_length)
            train_data = dset_cls(root=data_path, transform=train_transform)
            shape = train_data[0][0].unsqueeze(0).shape
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            n_train = len(train_data)
            split = n_train // 2
            indices = list(range(n_train))
            random.shuffle(indices)
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

            self.train_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=4,
                                                       pin_memory=True)
            self.valid_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       sampler=valid_sampler,
                                                       num_workers=4,
                                                       pin_memory=True)
        elif 'sport8' == self.args.train_dataset:
            dset_cls = dset.ImageFolder
            n_classes = 8
            data_path = '%s/Sport8/train' % self.args.data 
            val_path = '%s/Sport8/test' % self.args.data  
            train_transform, valid_transform = dutils.data_transforms_large(self.args.train_dataset,self.args.cutout_length)
            train_data = dset_cls(root=data_path, transform=train_transform)
            shape = train_data[0][0].unsqueeze(0).shape
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            n_train = len(train_data)
            split = n_train // 2
            indices = list(range(n_train))
            random.shuffle(indices)
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

            self.train_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=4,
                                                       pin_memory=True)
            self.valid_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       sampler=valid_sampler,
                                                       num_workers=4,
                                                       pin_memory=True)
        elif 'flowers102' == self.args.train_dataset:
            dset_cls = dset.ImageFolder
            n_classes = 102
            data_path = '%s/flowers102/train' % self.args.data
            val_path = '%s/flowers102/test' % self.args.data
            train_transform, valid_transform = dutils.data_transforms_large(self.args.train_dataset,self.args.cutout_length)
            train_data = dset_cls(root=data_path, transform=train_transform)
            shape = train_data[0][0].unsqueeze(0).shape
            assert shape[2] == shape[3], "not expected shape = {}".format(shape)
            n_train = len(train_data)
            split = n_train // 2
            indices = list(range(n_train))
            random.shuffle(indices)
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

            self.train_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       sampler=train_sampler,
                                                       num_workers=4,
                                                       pin_memory=True)
            self.valid_queue = torch.utils.data.DataLoader(train_data,
                                                       batch_size=self.args.batch_size,
                                                       sampler=valid_sampler,
                                                       num_workers=4,
                                                       pin_memory=True)                     
        elif 'food101' == self.args.train_dataset:
            traindir = os.path.join(self.args.data, 'train')
            validdir = os.path.join(self.args.data, 'val')
            train_transform, valid_transform = dutils.data_transforms_food101()
            train_data = dset.ImageFolder(
                traindir, train_transform)
            valid_data = dset.ImageFolder(
                validdir, valid_transform)

            self.train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

            self.valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        elif 'tiny-imagenet' == self.args.train_dataset:
            train_transform, valid_transform = dutils.data_transforms_tiny_imagenet()
            train_data = dartsdset.TinyImageNet200(self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dartsdset.TinyImageNet200(self.args.data, train=False, download=True,
                                                   transform=valid_transform)
            self.train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
            )
            self.valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
            )
    def _init_model(self):
        self.criterion = nn.CrossEntropyLoss()
        model = Network(self.args.image_channels, self.args.init_channels, self.args.train_classes,
                        layers=self.args.layers, criterion=self.criterion,
                        num_inp_node=2, num_meta_node=self.args.num_meta_node,
                        reduce_level=0 if 'cifar' in self.args.train_dataset else 3,
                        use_sparse=self.args.use_sparse)
        self.model = model.cuda()
        self.logger.info('param size = %fMB', dutils.calc_parameters_count(model))

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        last_epoch = -1 if self.args.start_epoch == 0 else self.args.start_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min, last_epoch=last_epoch
        )

        self.architect = Architecture(self.model, self.args)


    def run(self):
        self.logger.info('args = %s', self.args)
        run_start = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.scheduler.step()
            self.lr = self.scheduler.get_lr()[0]
            self.logger.info('epoch %d / %d lr %e', epoch, self.args.epochs, self.lr)

            # construct genotype and update topology graph
            genotype = self.model.genotype()

            self.logger.info('genotype = %s', genotype)

            print('alphas normal: \n', F.softmax(self.model.alphas_normal, dim=-1))
            print('alphas reduce: \n', F.softmax(self.model.alphas_reduce, dim=-1))

            # train and search the model
            train_acc, train_obj = self.train()

            # valid the model
            valid_acc, valid_obj = self.infer()
            self.logger.info('valid_acc %f', valid_acc)

            # save checkpoint
            dutils.save_checkpoint({
                'epoch': epoch + 1,
                'use_sparse': self.args.use_sparse,
                'state_dict': self.model.state_dict(),
                'dur_time': self.dur_time + time.time()-run_start,
                'arch_parameter': self.model._arch_parameters,
                'alphas_normal': self.model.alphas_normal,
                'alphas_reduce': self.model.alphas_reduce,
                'arch_optimizer': self.architect.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, is_best=False, save=args.save)
            self.logger.info('save checkpoint (epoch %d) in %s  dur_time: %s'
                             ,epoch, self.args.save, dutils.calc_time(self.dur_time + time.time()-run_start))
        with open(self.args.save + "/genotype.txt", "w") as f:
            f.write(str(genotype))

    def train(self):
        objs = dutils.AverageMeter()
        top1 = dutils.AverageMeter()

        for step, (input, target) in enumerate(self.train_queue):
            self.model.train()
            n = input.size(0)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # Get a random minibatch from the search queue(validation set) with replacement
            input_search, target_search = next(iter(self.valid_queue))
            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            # Update the architecture parameters
            self.architect.step(input, target, input_search, target_search, self.lr,
                           self.optimizer, unrolled=self.args.sec_approx)

            self.optimizer.zero_grad()

            logits = self.model(input)
            loss = self.criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            # Update the network parameters
            self.optimizer.step()

            prec1 = dutils.accuracy(logits, target, topk=(1,))[0]
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                self.logger.info('model size: %f', dutils.calc_parameters_count(self.model))
                self.logger.info('train %03d loss: %e top1: %f', step, objs.avg, top1.avg)

        return top1.avg, objs.avg

    def infer(self):
        objs = dutils.AverageMeter()
        top1 = dutils.AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_queue):
                input = input.cuda()
                target = target.cuda(non_blocking=True)

                logits = self.model(input)
                loss = self.criterion(logits, target)

                prec1 = dutils.accuracy(logits, target, topk=(1,))[0]
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

                if step % args.report_freq == 0:
                    self.logger.info('valid %03d %e %f', step, objs.avg, top1.avg)

            return top1.avg, objs.avg

if __name__ == '__main__':
    args = parser.parse_args()
    search_network = SearchNetwork(args)
    search_network.run()
