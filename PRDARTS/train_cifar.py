import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import math
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from gpu_thread import GpuLogThread
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
# misc args
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='/tmp/pdarts/', help='experiment path')
parser.add_argument('--tmp_data_dir', type=str, default='~/Data/Datasets/cifar10_data/', help='temp data dir')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=["CIFAR10","CIFAR100","Sport8","MIT67","flowers102"], help='search with different datasets')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--test', type=str, default='false', help='debug printing, end cycles/epochs quickly')
# training args
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--batch_size_min', type=int, default=4, help='minimum batch size')
parser.add_argument('--batch_size_max', type=int, default=128, help='maximum batch size')
parser.add_argument('--batch_multiples', type=int, default=2, help='batch size multiples')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--auxiliary', type=str, default='false', help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', type=str, default='false', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
# model args
parser.add_argument('--arch', type=str, default='PR_DARTS_DL1', help='which architecture to use')
parser.add_argument('--arch_pref_sc', type=int, default=2, help='pref num of skip connections, replaces %d in arch')  # unnecessary remainder of PDARTS influenced experiments
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')

args, unparsed = parser.parse_known_args()
args.test = args.test.lower().startswith('t')
args.auxiliary = args.auxiliary.lower().startswith('t')
args.cutout = args.cutout.lower().startswith('t')
#args.cifar100 = args.cifar100.lower().startswith('t')

# args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d.%m.%y %H:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save) #flush_secs=30)


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    gpu_logger = GpuLogThread(list(range(num_gpus)), writer, seconds=10 if args.test else 300)
    gpu_logger.start()

    genotype = genotypes.load_genotype(args.arch, skip_cons=args.arch_pref_sc)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    if args.dataset == "CIFAR100":
        CLASSES = 100
    elif args.dataset == "CIFAR10":
        CLASSES = 10
    elif args.dataset == 'MIT67':
        dset_cls = dset.ImageFolder
        CLASSES = 67
    elif args.dataset == 'Sport8':
        dset_cls = dset.ImageFolder
        CLASSES = 8
    elif args.dataset == "flowers102":
        dset_cls = dset.ImageFolder
        CLASSES = 102
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, largemode=True if args.dataset in utils.LARGE_DATASETS else False)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    logging.info("param count = %d", utils.count_parameters(model))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    train_transform, valid_transform = utils.data_transforms(args.dataset, args.cutout, args.cutout_length)
    if args.dataset == "CIFAR100":
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    elif args.dataset == "CIFAR10":
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    elif args.dataset == 'MIT67':
        dset_cls = dset.ImageFolder
        data_path = '%s/MIT67/train' % args.tmp_data_dir  # 'data/MIT67/train'
        val_path = '%s/MIT67/test' % args.tmp_data_dir  # 'data/MIT67/val'
        train_data = dset_cls(root=data_path, transform=train_transform)
        valid_data = dset_cls(root=val_path, transform=valid_transform)
    elif args.dataset == 'Sport8':
        dset_cls = dset.ImageFolder
        data_path = '%s/Sport8/train' % args.tmp_data_dir  # 'data/Sport8/train'
        val_path = '%s/Sport8/test' % args.tmp_data_dir  # 'data/Sport8/val'
        train_data = dset_cls(root=data_path, transform=train_transform)
        valid_data = dset_cls(root=val_path, transform=valid_transform)
    elif args.dataset == "flowers102":
        dset_cls = dset.ImageFolder
        data_path = '%s/flowers102/train' % args.tmp_data_dir
        val_path = '%s/flowers102/test' % args.tmp_data_dir
        train_data = dset_cls(root=data_path, transform=train_transform)
        valid_data = dset_cls(root=val_path, transform=valid_transform)

    train_iterator = utils.DynamicBatchSizeLoader(torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_multiples, shuffle=True, pin_memory=True, num_workers=args.workers),
        args.batch_size_min)
    test_iterator = utils.DynamicBatchSizeLoader(torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_multiples, shuffle=False, pin_memory=True, num_workers=args.workers),
        args.batch_size_min)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = 0.0
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        drop_path_prob = args.drop_path_prob * epoch / args.epochs
        logging.info('Epoch: %d lr %e', epoch, lr)
        if num_gpus > 1:
            model.module.drop_path_prob = drop_path_prob
        else:
            model.drop_path_prob = drop_path_prob
        epoch_start_time = time.time()
        train_acc, train_obj = train(train_iterator, test_iterator, model, criterion, optimizer, gpu_logger)
        logging.info('Train_acc: %f', train_acc)

        test_acc, test_obj = infer(test_iterator, model, criterion)
        if test_acc > best_acc:
            best_acc = test_acc
        logging.info('Valid_acc: %f', test_acc)
        epoch_duration = time.time() - epoch_start_time
        utils.save(model, os.path.join(args.save, 'weights.pt'))

        # log info
        print('Epoch time: %ds.' % epoch_duration)
        writer.add_scalar('epoch/lr', lr, epoch)
        writer.add_scalar('epoch/drop_path_prob', drop_path_prob, epoch)
        writer.add_scalar('epoch/seconds', epoch_duration, epoch)
        writer.add_scalar('epoch/batch_size', train_iterator.batch_size, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('train/loss', train_obj, epoch)
        writer.add_scalar('test/accuracy', test_acc, epoch)
        writer.add_scalar('test/loss', test_obj, epoch)

        scheduler.step()
    gpu_logger.stop()


def train(train_iterator, test_iterator, model, criterion, optimizer, gpu_logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    inc_batch_size = args.batch_size_max > args.batch_size_min
    for step in train_iterator.yield_steps():
        input_, target = train_iterator.__next__()
        input_ = input_.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input_)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input_.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

        gpu_logger.wakeup()
        # if there is gpu memory free, try to scale the batch size up
        if inc_batch_size and (step + 1) % 10 == 0:
            usage = gpu_logger.get_highest_recent()
            new_batch_size = math.floor(0.95 * train_iterator.batch_size / usage)
            new_batch_size -= new_batch_size % args.batch_multiples
            new_batch_size = min(new_batch_size, args.batch_size_max, 2*train_iterator.batch_size)
            if new_batch_size > train_iterator.batch_size:
                logging.info('Set new batch size from %d to %d, cur usage is %.2f (step %d)' %
                             (train_iterator.batch_size, new_batch_size, usage, gpu_logger.step_recent))
                train_iterator.set_batch_size(new_batch_size)
                test_iterator.set_batch_size(new_batch_size)
                torch.cuda.empty_cache()

    return top1.avg, objs.avg


def infer(test_iterator, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step in test_iterator.yield_steps():
        input_, target = test_iterator.__next__()
        input_ = input_.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input_)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input_.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Test Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
