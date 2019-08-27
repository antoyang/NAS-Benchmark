import os
import sys
import glob
import time
import copy
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch import Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNetworkImageNet

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data/imagenet')
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False) # best practice: do not use lazy_load. when using zip_file, do not use lazy_load
parser.add_argument('--dataset', type=str, default = "Imagenet")
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=48)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--drop_path_keep_prob', type=float, default=1.0)
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--l2_reg', type=float, default=3e-5)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def train(train_queue, model, optimizer, global_step, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()
    
        optimizer.zero_grad()
        logits, aux_logits = model(input, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
    
        if (step+1) % 100 == 0:
            logging.info('train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, global_step


def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()
        
            logits, _ = model(input)
            loss = criterion(logits, target)
        
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
        
            if (step+1) % 100 == 0:
                logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        traindir = os.path.join(args.data, 'train.zip')
        validdir = os.path.join(args.data, 'valid.zip')
        if args.lazy_load:
            train_data = utils.ZipDataset(traindir, train_transform)
            valid_data = utils.ZipDataset(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=32)
            valid_data = utils.InMemoryZipDataset(validdir, valid_transform, num_workers=32)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'test') #valid
        if args.lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
            valid_data = dset.ImageFolder(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=32)
            valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=32)
    
    logging.info('Found %d in training data', len(train_data))
    logging.info('Found %d in validation data', len(valid_data))

    args.steps = int(np.ceil(len(train_data) / (args.batch_size))) * torch.cuda.device_count() * args.epochs

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    if args.dataset == "sport8":
        model = NASNetworkImageNet(args, 8, args.layers, args.nodes, args.channels, args.keep_prob,
                       args.drop_path_keep_prob, args.use_aux_head, args.steps, args.arch)
    elif args.dataset == "mit67":
        model = NASNetworkImageNet(args, 67, args.layers, args.nodes, args.channels, args.keep_prob,
                       args.drop_path_keep_prob, args.use_aux_head, args.steps, args.arch)
    elif args.dataset == 'flowers102':
        model = NASNetworkImageNet(args, 102, args.layers, args.nodes, args.channels, args.keep_prob,
                                   args.drop_path_keep_prob, args.use_aux_head, args.steps, args.arch)
    else:
        model = NASNetworkImageNet(args, 1000, args.layers, args.nodes, args.channels, args.keep_prob,
                                   args.drop_path_keep_prob, args.use_aux_head, args.steps, args.arch)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("multi adds = %fM", model.multi_adds / 1000000)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()
    """if args.dataset == "sport8":
        train_criterion = CrossEntropyLabelSmooth(8, args.label_smooth).cuda()
    elif args.dataset == "mit67":
        train_criterion = CrossEntropyLabelSmooth(67, args.label_smooth).cuda()
    else:
        train_criterion = CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()"""
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        #args.lr,
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, args.gamma, epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    
    logging.info("Args = %s", args)
    
    _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.output_dir)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_imagenet(model_state_dict, optimizer_state_dict, epoch=epoch-1)

    while epoch < args.epochs:
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_acc, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
        logging.info('train_acc %f', train_acc)
        valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)

        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save(args.output_dir, args, model, epoch, step, optimizer, best_acc_top1, is_best)
        

if __name__ == '__main__':
    main()
