""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc

LARGE_DATASETS = ["imagenet", "mit67", "sport8", "flowers102"]

def get_data(dataset, data_root, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()
    
    data_path = data_root
    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    if dataset == 'cifar100':
        dset_cls = dset.CIFAR100
        n_classes = 100
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    #New Datasets
    elif dataset == 'mit67':
        dset_cls = dset.ImageFolder
        n_classes = 67
        data_path = '%s/MIT67/train' % data_root  # 'data/MIT67/train'
        val_path = '%s/MIT67/test' % data_root  # 'data/MIT67/val'
    elif dataset == 'sport8':
        dset_cls = dset.ImageFolder
        n_classes = 8
        data_path = '%s/Sport8/train' % data_root  # 'data/Sport8/train'
        val_path = '%s/Sport8/test' % data_root # 'data/Sport8/val'
    elif dataset == 'flowers102':
        dset_cls = dset.ImageFolder
        n_classes = 102
        data_path = '%s/flowers102/train' % data_root  # 'data/flowers102/train'
        val_path = '%s/flowers102/test' % data_root  # 'data/flowers102/val' 
    
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    if dataset in LARGE_DATASETS:
        trn_data = dset_cls(root=data_path, transform=trn_transform)
        shape = trn_data[0][0].unsqueeze(0).shape
        print(shape)
        assert shape[2] == shape[3], "not expected shape = {}".format(shape)
        input_size = shape[2]
    else:
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
        # assuming shape is NHW or NHWC
        try:
            shape = trn_data.data.shape
        except AttributeError:
            shape = trn_data.train_data.shape
        assert shape[1] == shape[2], "not expected shape = {}".format(shape)
        input_size = shape[1]
        
    input_channels = 3 if len(shape) == 4 else 1
    # print("Number of input channels: ", input_channels)

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        if dataset in LARGE_DATASETS:
            ret.append(dset_cls(root=val_path, transform=val_transform))
        else:
            ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
