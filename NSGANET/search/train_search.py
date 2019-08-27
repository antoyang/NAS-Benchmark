import sys
# update your project root path before running
sys.path.insert(0, '/cache/NSGANET')

import os
import random
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.macro_models import EvoNetwork
from models.micro_models import NetworkCIFAR, NetworkImageNet

import search.cifar10_search as my_cifar10

import time
from misc import utils
from search import micro_encoding
from search import macro_encoding
from misc.flops_counter import add_flops_counting_methods


device = 'cuda'


def main(genome, epochs, search_space='micro',
         save='Design_1', expr_root='search', seed=0, gpu=0, init_channels=24,
         layers=11, auxiliary=False, cutout=False, drop_path_prob=0.0,data_path="../data", dataset="CIFAR10"):

    # ---- train logger ----------------- #
    save_pth = os.path.join(expr_root, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    # ---- parameter values setting ----- #
    if dataset == "CIFAR10":
        CLASSES = 10
    elif dataset == "CIFAR100":
        CLASSES = 100
    elif dataset == "Sport8":
        CLASSES = 8
    elif dataset == "MIT67":
        CLASSES = 67
    elif dataset == "flowers102":
        CLASSES = 102
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    data_root = data_path
    batch_size = 128
    cutout_length = 16
    auxiliary_weight = 0.4
    grad_clip = 5
    report_freq = 50
    train_params = {
        'auxiliary': auxiliary,
        'auxiliary_weight': auxiliary_weight,
        'grad_clip': grad_clip,
        'report_freq': report_freq,
    }

    if search_space == 'micro':
        genotype = micro_encoding.decode(genome)
        if dataset=="CIFAR10" or dataset == "CIFAR100":
            model = NetworkCIFAR(init_channels, CLASSES, layers, auxiliary, genotype)
        else:
            model = NetworkImageNet(init_channels, CLASSES, layers, auxiliary, genotype)
    elif search_space == 'macro':
        genotype = macro_encoding.decode(genome)
        channels = [(3, init_channels),
                    (init_channels, 2*init_channels),
                    (2*init_channels, 4*init_channels)]
        model = EvoNetwork(genotype, channels, CLASSES, (32, 32), decoder='residual')
    else:
        raise NameError('Unknown search space type')

    # logging.info("Genome = %s", genome)
    logging.info("Architecture = %s", genotype)

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) / 1e6)
    model = model.to(device)

    logging.info("param size = %fMB", n_params)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    if dataset == "CIFAR10" or dataset == "CIFAR100":
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        if cutout:
            train_transform.transforms.append(utils.Cutout(cutout_length))

        train_transform.transforms.append(transforms.Normalize(MEAN, STD))

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    if dataset == "CIFAR10":
        train_data = my_cifar10.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        valid_data = my_cifar10.CIFAR10(root=data_root, train=True, download=True, transform=valid_transform)#dunno
    elif dataset == "CIFAR100":
        train_data = dset.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=data_root, train=True, download=True, transform=valid_transform)
    else:
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
        valid_transform = transforms.Compose(transf_val + normalize)
        if cutout:
            train_transform.transforms.append(utils.Cutout(cutout_length))

        train_data = dset.ImageFolder(root=data_path+"/"+dataset+"/train", transform=train_transform)
        valid_data = dset.ImageFolder(root=data_path+"/"+dataset+"/test", transform=valid_transform)

    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:n_train]),
        pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.droprate = drop_path_prob * epoch / epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, train_params)
        logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # calculate for flops
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 3, 32, 32) #to change
    model(torch.autograd.Variable(random_data).to(device))
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    logging.info('flops = %f', n_flops)

    # save to file
    # os.remove(os.path.join(save_pth, 'log.txt'))
    with open(os.path.join(save_pth, 'log.txt'), "w") as file:
        file.write("Genome = {}\n".format(genome))
        file.write("Architecture = {}\n".format(genotype))
        file.write("param size = {}MB\n".format(n_params))
        file.write("flops = {}MB\n".format(n_flops))
        file.write("valid_acc = {}\n".format(valid_acc))
    # logging.info("Architecture = %s", genotype))
    with open(os.path.join(save_pth, 'genotype.txt'), "w") as f:
        f.write(str(genotype))
    return {
        'valid_acc': valid_acc,
        'params': n_params,
        'flops': n_flops,
    }

# Training
def train(train_queue, net, criterion, optimizer, params):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
        loss = criterion(outputs, targets)

        if params['auxiliary']:
            loss_aux = criterion(outputs_aux, targets)
            loss += params['auxiliary_weight'] * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #     if step % args.report_freq == 0:
    #         logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)
    #
    # logging.info('train acc %f', 100. * correct / total)

    return 100.*correct/total, train_loss/total

def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc, test_loss/total


if __name__ == "__main__":
    DARTS_V2 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]], [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
               [[[0, 0], [0, 1]], [[2, 2], [0, 1]], [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    start = time.time()
    print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_16', seed=1, init_channels=16,
               auxiliary=False, cutout=False, drop_path_prob=0.0,data_path="../data"))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
    # start = time.time()
    # print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_32', seed=1, init_channels=32))
    # print('Time elapsed = {} mins'.format((time.time() - start) / 60))

