from __future__ import print_function
import os
import sys
import errno
import os.path
import subprocess
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms

class AverageMeter(object):
    """Compute the average meter"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class Cutout(object):
    """Cotout the training data

    Attributes:
        length: int, the length to clip
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h,w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""

    rst = subprocess.run('nvidia-smi -q -d Memory',stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    rst = rst.strip().split('\n')
    memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
    id = int(np.argmax(memory_available))
    return id, memory_available

def concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def accuracy(output, target, topk=(1,)):
    """Compute the top1 and top5 accuracy

    Args:
        output: torch.tensor, the output of model, likes logits
        target: torch.tensor, the target the input data
        topk: tuple, topk tuple,likes (1,5)), indicate get accuracy top1 and top5

    return

    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Return the k largest elements of the given input tensor
    # along a given dimension -> N * k
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))

    return res

def data_transforms_cifar(args):
    """
    Args:
        args:
            cutout: boolean, whether cutout the data
            cutout_length: int, the length of cutout
    Returns:
    """
    normalize = transforms.Normalize(mean = [0.491, 0.482, 0.44653124],std = [0.247, 0.243, 0.261])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

def data_transforms_tiny_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, valid_transform

def data_transforms_imagenet():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                normalize,
            ])

    return train_transform, valid_transform

def data_transforms_large(dataset, cutout_length):
    dataset = dataset.lower()
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transf_train = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2)
    ]
    transf_val = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    train_transform = transforms.Compose(transf_train + normalize)
    valid_transform = transforms.Compose(
        transf_val + normalize)  # FIXME validation is not set to square proportions, is this an issue?

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform

def data_transforms_food101():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128), # default bilinear for interpolation
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                normalize,
            ])

    return train_transform, valid_transform

def calc_parameters_count(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day':t, 'hour':h, 'minute':m, 'second':int(s)}

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.tensor(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir: {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr