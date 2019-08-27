import os
import sys
import time
import random
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

from data.data import get_loaders
from torch.autograd import Variable
from micro_child import CNN
from micro_controller import Controller


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--fixed_arc', type=str, default=None, help='architecture')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--child_lr_max', type=float, default=0.05)
parser.add_argument('--child_lr_min', type=float, default=0.0005)
parser.add_argument('--child_lr_T_0', type=int, default=10)
parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_num_layers', type=int, default=6)
parser.add_argument('--child_out_filters', type=int, default=20)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_num_cells', type=int, default=5)
parser.add_argument('--child_use_aux_heads', type=bool, default=False)

parser.add_argument('--controller_lr', type=float, default=0.0035)
parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)

parser.add_argument('--lstm_size', type=int, default=64)
parser.add_argument('--lstm_num_layers', type=int, default=1)
parser.add_argument('--lstm_keep_prob', type=float, default=0)
parser.add_argument('--temperature', type=float, default=5.0)

parser.add_argument('--entropy_weight', type=float, default=0.0001)
parser.add_argument('--bl_dec', type=float, default=0.99)

args = parser.parse_args()

#args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == "CIFAR10":
    CLASSES = 10
elif args.dataset == "MIT67":
    CLASSES = 67
elif args.dataset == "Sport8":
    CLASSES = 8
elif args.dataset == "flowers102":
    CLASSES = 102

baseline = None
epoch = 0

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model = CNN(args)
    model.cuda()

    controller = Controller(args)
    controller.cuda()
    baseline = None

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        args.controller_lr,
        betas=(0.1,0.999),
        eps=1e-3,
    )

    train_loader, reward_loader, valid_loader = get_loaders(args)

    scheduler = utils.LRScheduler(optimizer, args)

    for epoch in range(args.epochs):
        lr = scheduler.update(epoch)
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc = train(train_loader, model, controller, optimizer)
        logging.info('train_acc %f', train_acc)

        train_controller(reward_loader, model, controller, controller_optimizer)

        # validation
        valid_acc = infer(valid_loader, model, controller)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_loader, model, controller, optimizer):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()

    for step, (data, target) in enumerate(train_loader):
        model.train()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        controller.eval()
        dag, _, _ = controller()

        logits, _ = model(data, dag)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, total_loss.avg, total_top1.avg)

    return total_top1.avg

def train_controller(reward_loader, model, controller, controller_optimizer):
    global baseline
    total_loss = utils.AvgrageMeter()
    total_reward = utils.AvgrageMeter()
    total_entropy = utils.AvgrageMeter()

    #for step, (data, target) in enumerate(reward_loader):
    for step in range(300):
        data, target = reward_loader.next_batch()
        model.eval()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        controller_optimizer.zero_grad()

        controller.train()
        dag, log_prob, entropy = controller()

        with torch.no_grad():
            logits, _ = model(data, dag)
            reward = utils.accuracy(logits, target)[0]

        if args.entropy_weight is not None:
            reward += args.entropy_weight*entropy

        log_prob = torch.sum(log_prob)
        if baseline is None:
            baseline = reward
        baseline -= (1 - args.bl_dec) * (baseline - reward)

        loss = log_prob * (reward - baseline)
        loss = loss.sum()

        loss.backward()

        controller_optimizer.step()

        total_loss.update(loss.item(), n)
        total_reward.update(reward.item(), n)
        total_entropy.update(entropy.item(), n)

        if step % args.report_freq == 0:
            #logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
            logging.info('controller %03d %e %f %f', step, total_loss.avg, total_reward.avg, baseline.item())
            #tensorboard.add_scalar('controller/loss', loss, epoch)
            #tensorboard.add_scalar('controller/reward', reward, epoch)
            #tensorboard.add_scalar('controller/entropy', entropy, epoch)

def infer(valid_loader, model, controller):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.eval()
    controller.eval()

    with torch.no_grad():
        for step in range(10):
            data, target = valid_loader.next_batch()
            data = data.cuda()
            target = target.cuda()

            dag, _, _ = controller()

            logits, _ = model(data, dag)
            loss = F.cross_entropy(logits, target)

            prec1 = utils.accuracy(logits, target)[0]
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            #if step % args.report_freq == 0:
            logging.info('valid %03d %e %f', step, loss.item(), prec1.item())
            logging.info('normal cell %s', str(dag[0]))
            logging.info('reduce cell %s', str(dag[1]))

    return total_top1.avg


if __name__ == '__main__':
    main() 

