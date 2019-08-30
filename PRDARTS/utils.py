import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
import logging
from torch.autograd import Variable

LARGE_DATASETS = ["Sport8", "MIT67", "flowers102"]

class AvgrageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
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


def data_transforms_cifar(args, mean, std):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, valid_transform


def data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    return data_transforms_cifar(args, CIFAR_MEAN, CIFAR_STD)


def data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]
    return data_transforms_cifar(args, CIFAR_MEAN, CIFAR_STD)


def data_transforms(dataset, cutout, cutout_length):
    if dataset in LARGE_DATASETS:
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
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose(transf_val + normalize)
        return train_transform, valid_transform
    if dataset == "CIFAR10":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        return train_transform, valid_transform
    if dataset == "CIFAR100":
        CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
        CIFAR_STD = [0.2675, 0.2565, 0.2761]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        if cutout:
            train_transform.transforms.append(Cutout(cutout_length))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        return train_transform, valid_transform


def count_parameters(model):
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'auxiliary' not in n)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path, strict=False):
    if not os.path.isfile(model_path):
        logging.warning('Failed loading model params from: %s, does not exist' % model_path)
        return None
    logging.info('Loaded (some) model params from: %s' % model_path)
    return model.load_state_dict(torch.load(model_path), strict=strict)


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None, clean=True):
    os.makedirs(path, exist_ok=True)
    if clean:
        for file in os.listdir(path):
            (os.remove if os.path.isfile(path+file) else shutil.rmtree)(path+file)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class InfIterator:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None

    def __next__(self):
        while True:
            try:
                if self.iterator is None:
                    self.iterator = iter(self.loader)
                return self.iterator.__next__()
            except:
                del self.iterator
                self.iterator = None

    def __len__(self):
        return len(self.loader)


class DynamicBatchSizeLoader:
    def __init__(self, loader, batch_size):
        self.iterator = InfIterator(loader)
        self.loader_batch_size = loader.batch_size
        self.batch_size = batch_size

    def set_batch_size(self, batch_size):
        d, m = divmod(batch_size, self.loader_batch_size)
        self.batch_size = int(d*self.loader_batch_size)
        if m % self.loader_batch_size > 0:
            logging.warning('batch size %d is not multiple of loader batch size %d, using %d instead' %
                            (batch_size, self.loader_batch_size, self.batch_size))

    def __next__(self):
        inputs_, labels = [], []
        for _ in range(self.batch_size // self.loader_batch_size):
            in_, l_ = self.iterator.__next__()
            inputs_.append(in_)
            labels.append(l_)
        return torch.cat(inputs_, dim=0), torch.cat(labels, dim=0)

    def __len__(self):
        return int((self.loader_batch_size / self.batch_size) * len(self.iterator))

    def yield_steps(self):
        """ yields consecutive ints, for one epoch, if __next__() is called each step (fixes changing batch sizes) """
        stepped = 0
        for step in range(len(self.iterator)):
            stepped += self.batch_size / self.loader_batch_size
            if stepped > len(self.iterator):
                break
            yield step


def op_similarity(ops: list) -> float:
    """
    for a list of operations, return the average of how similar each operation is with all others
    they are not similar, 0, if they are different types of operations, otherwise up to 1 (exact same kwargs)
    """
    def similarity(op1, op2):
        n1, kw1 = op1
        n2, kw2 = op2
        # same type of op?
        if n1 != n2:
            return 0
        # op kwargs about the same?
        k_sims = [1]
        for k in kw1.keys():
            s, v1, v2 = 0, kw1[k], kw2[k]
            if isinstance(v1, list):
                n = min(len(v1), len(v2)) + 1
                s += 1/n if len(v1) == len(v2) else 0
                for v in range(n-1):
                    s += 1 / n if v1[v] == v2[v] else 0
            else:
                s = 1 if v1 == v2 else 0
            k_sims.append(s)
        return sum(k_sims) / len(k_sims)

    sims = []
    for i in range(len(ops)):
        sims.append(1)
        for j in range(i+1, len(ops)):
            sm = similarity(ops[i], ops[j])
            sims.append(sm)     # assume that the similarity measure is commutative
            sims.append(sm)
    if len(sims) == 0:
        return None
    return sum(sims) / len(sims)
