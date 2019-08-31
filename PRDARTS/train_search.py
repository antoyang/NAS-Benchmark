import os
import sys
import time
import glob
import copy
import numpy as np
import torch
import utils
import logging
import argparse
import math
from collections import namedtuple
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from gpu_thread import GpuLogThread
from model_search import Network
from genotypes import Genotype, get_init_ops, save_genotype
from operations import get_morphed_kwargs, op_as_str, get_morph_restrictions

Cycle = namedtuple('Cycle', ['num', 'net_layers', 'net_init_c', 'net_dropout', 'ops_keep', 'load', 'init_morphed',
                             'epochs', 'grace_epochs', 'morphs', 'is_last'])

parser = argparse.ArgumentParser("cifar")
# misc args
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--save', type=str, default='/', help='experiment path')
parser.add_argument('--tmp_data_dir', type=str, default='~/Data/Datasets/cifar10_data/', help='temp data dir')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=["CIFAR10","CIFAR100","Sport8","MIT67","flowers102"], help='search with different datasets')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--test', type=str, default='true', help='debug printing, end cycles/epochs quickly')
parser.add_argument('--test_batches', type=int, default=5, help='how many batches in case of args.test')
# training args
parser.add_argument('--batch_size_min', type=int, default=4, help='minimum batch size')
parser.add_argument('--batch_size_max', type=int, default=128, help='maximum batch size')
parser.add_argument('--batch_multiples', type=int, default=2, help='batch size multiples')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--cutout', type=str, default='false', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# model args
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--blocks', type=int, default=4, help='number of blocks per cell')
# search args
parser.add_argument('--primitives', type=str, default='prdarts4', help='which set of primitives (operations) to use')
parser.add_argument('--epochs', type=str,        default='1,   1,   1  ', help='num of training epochs')
parser.add_argument('--grace_epochs', type=str,  default='0,   0,   0  ', help='grace epochs without arc training')
parser.add_argument('--dropout_rate', type=str,  default='0.0, 0.0, 0.0', help='dropout rate of skip connect')
parser.add_argument('--add_width', type=str,     default='0,   0,   0  ', help='add channels')
parser.add_argument('--add_layers', type=str,    default='0,   0,   0  ', help='add layers')
parser.add_argument('--num_to_keep', type=str,   default='5,   4,   1  ', help='how many paths/op to keep after a cycle')
parser.add_argument('--num_morphs', type=str,    default='1,   0,   0  ', help='how many paths/op to morph after a cycle')
parser.add_argument('--try_load', type=str, default='true', help='try load params from previous cycles')
parser.add_argument('--reset_alphas', type=str, default='true', help='reset arc params after loading')
parser.add_argument('--restrict_zero', type=str, default='false', help='prevent zero cons in the final genotype')
parser.add_argument('--morph_restrictions', type=str, default='r_depth', help='how to restrict op morphisms')

args = parser.parse_args()
args.cutout = args.cutout.lower().startswith('t')
#args.cifar100 = args.cifar100.lower().startswith('t')
args.test = args.test.lower().startswith('t')
args.try_load = args.try_load.lower().startswith('t')
args.reset_alphas = args.reset_alphas.lower().startswith('t')
args.restrict_zero = args.restrict_zero.lower().startswith('t')
args.epochs = [int(n) for n in str(args.epochs).replace(' ', '').split(',')]
args.grace_epochs = [int(n) for n in str(args.grace_epochs).replace(' ', '').split(',')]
args.dropout_rate = [float(n) for n in str(args.dropout_rate).replace(' ', '').split(',')]
args.add_width = [int(n) for n in str(args.add_width).replace(' ', '').split(',')]
args.add_layers = [int(n) for n in str(args.add_layers).replace(' ', '').split(',')]
args.num_to_keep = [int(n) for n in str(args.num_to_keep).replace(' ', '').split(',')]
args.num_morphs = [int(n) for n in str(args.num_morphs).replace(' ', '').split(',')]
# args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))

PRIMITIVES = get_init_ops(args.primitives)
morph_restrictions = get_morph_restrictions(args.morph_restrictions)
#utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = "[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s" if args.test else '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.test else logging.INFO,
                    format=log_format, datefmt='%d.%m.%y %H:%M:%S')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save) #flush_secs=30)
model_path = os.path.join(args.save, 'weights.pt')


def parse_cycles():
    logging.debug(locals())
    assert len(args.add_width) == len(args.add_layers) == len(args.dropout_rate) == len(args.num_to_keep)
    assert len(args.add_width) == len(args.num_morphs) == len(args.grace_epochs) == len(args.epochs)
    cycles = []
    for i in range(len(args.add_width)):
        try_load = args.try_load and i > 0
        net_layers = args.layers + int(args.add_layers[i])
        net_init_c = args.init_channels + int(args.add_width[i])
        if len(cycles) > 0 and try_load:
            if cycles[-1].net_layers != net_layers or cycles[-1].net_init_c != net_init_c:
                try_load = False
        cycles.append(Cycle(
            num=i,
            net_layers=args.layers + int(args.add_layers[i]),
            net_init_c=args.init_channels + int(args.add_width[i]),
            net_dropout=float(args.dropout_rate[i]),
            ops_keep=args.num_to_keep[i],
            epochs=args.epochs[i],
            grace_epochs=args.grace_epochs[i] if not args.test else 0,
            morphs=args.num_morphs[i],
            init_morphed=try_load,
            load=try_load,
            is_last=(i == len(args.num_to_keep) - 1)))
    return cycles


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)
    gpu_logger = GpuLogThread([args.gpu], writer, seconds=15 if not args.test else 1)
    gpu_logger.start()
    logging.debug(locals())
    model = None

    # prepare dataset
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

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    import random;random.shuffle(indices)

    train_iterator = utils.DynamicBatchSizeLoader(torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_multiples,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers), args.batch_size_min)

    valid_iterator = utils.DynamicBatchSizeLoader(torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_multiples,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers), args.batch_size_min)

    # build Network
    logging.debug('building network')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    num_graph_edges = sum(list(range(2, 2 + args.blocks)))
    switches_normal = SwitchManager(num_graph_edges, copy.deepcopy(PRIMITIVES), 'normal')
    switches_reduce = SwitchManager(num_graph_edges, copy.deepcopy(PRIMITIVES), 'reduce')

    total_epochs = 0
    for cycle in parse_cycles():
        logging.debug('new cycle %s' % repr(cycle))
        print('\n' * 3, '-' * 100)
        print(cycle)
        print('', '-' * 100, '\n')

        writer.add_scalar('cycle/net_layers', cycle.net_layers, cycle.num)
        writer.add_scalar('cycle/net_init_c', cycle.net_init_c, cycle.num)
        writer.add_scalar('cycle/net_dropout', cycle.net_dropout, cycle.num)
        writer.add_scalar('cycle/ops_keep', cycle.ops_keep, cycle.num)
        writer.add_scalar('cycle/epochs', cycle.epochs, cycle.num)
        writer.add_scalar('cycle/grace_epochs', cycle.grace_epochs, cycle.num)
        writer.add_scalar('cycle/morphs', cycle.morphs, cycle.num)
        switches_normal.plot_ops(logging.info, writer, cycle.num)
        switches_reduce.plot_ops(logging.info, writer, cycle.num)

        # rebuild the model in each cycle, clean up the cache...
        logging.debug('building model')
        del model
        torch.cuda.empty_cache()
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
        model = Network(cycle.net_init_c,
                        CLASSES,
                        cycle.net_layers,
                        criterion,
                        switches_normal=switches_normal,
                        switches_reduce=switches_reduce,
                        steps=args.blocks,
                        p=cycle.net_dropout,
                        largemode=True if args.dataset in utils.LARGE_DATASETS else False)
        gpu_logger.reset_recent()
        if cycle.load:
            utils.load(model, model_path)
            if args.reset_alphas:
                model.reset_alphas()
        if args.test:
            model.randomize_alphas()
        if cycle.init_morphed:
            model.init_morphed(switches_normal, switches_reduce)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        logging.debug('building optimizers')
        optimizer = torch.optim.SGD(model.net_parameters,
                                    args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.arch_parameters,
                                       lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                       weight_decay=args.arch_weight_decay)
        logging.debug('building scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cycle.epochs),
                                                               eta_min=args.learning_rate_min)

        if args.batch_size_max > args.batch_size_min:
            train_iterator.set_batch_size(args.batch_size_min)
            valid_iterator.set_batch_size(args.batch_size_min)

        sm_dim = -1
        scale_factor = 0.2
        for epoch in range(cycle.epochs):
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < cycle.grace_epochs:
                model.update_p(cycle.net_dropout * (cycle.epochs - epoch - 1) / cycle.epochs)
            else:
                model.update_p(cycle.net_dropout * np.exp(-(epoch - cycle.grace_epochs) * scale_factor))
            train_acc, train_obj = train(train_iterator, valid_iterator, model, criterion, optimizer, optimizer_a,
                                         gpu_logger, train_arch=epoch >= cycle.grace_epochs)
            epoch_duration = time.time() - epoch_start

            # log info
            logging.info('Train_acc %f', train_acc)
            logging.info('Epoch time: %ds', epoch_duration)
            writer.add_scalar('train/accuracy', train_acc, total_epochs)
            writer.add_scalar('train/loss', train_obj, total_epochs)
            writer.add_scalar('epoch/lr', lr, total_epochs)
            writer.add_scalar('epoch/seconds', epoch_duration, total_epochs)
            writer.add_scalar('epoch/model.p', model.p, total_epochs)
            writer.add_scalar('epoch/batch_size', train_iterator.batch_size, total_epochs)

            # validation, only for the last 5 epochs in a cycle
            if cycle.epochs - epoch < 5:
                valid_acc, valid_obj = infer(valid_iterator, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
                writer.add_scalar('valid/accuracy', valid_acc, total_epochs)
                writer.add_scalar('valid/loss', valid_obj, total_epochs)

            total_epochs += 1
            gpu_logger.reset_recent()
            scheduler.step()

        utils.save(model, model_path)

        print('\n' * 2, '------Dropping/morphing paths------')
        # Save switches info for s-c refinement.
        if cycle.is_last:
            switches_normal_copy = switches_normal.copy()
            switches_reduce_copy = switches_reduce.copy()

        # drop operations with low architecture weights, add morphed ones
        arch_param = model.arch_parameters
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
        switches_normal.drop_and_morph(normal_prob, cycle.ops_keep, writer, cycle.num, num_morphs=cycle.morphs,
                                       no_zero=cycle.is_last and args.restrict_zero, keep_morphable=not cycle.is_last)
        reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
        switches_reduce.drop_and_morph(reduce_prob, cycle.ops_keep, writer, cycle.num, num_morphs=cycle.morphs,
                                       no_zero=cycle.is_last and args.restrict_zero, keep_morphable=not cycle.is_last)
        logging.info('switches_normal = \n%s', switches_normal)
        logging.info('switches_reduce = \n%s', switches_reduce)

        # end last cycle with shortcut/zero pruning and save the genotype
        if cycle.is_last:
            #import ipdb;ipdb.set_trace()
            arch_param = model.arch_parameters
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for _ in range(num_graph_edges)]
            reduce_final = [0 for _ in range(num_graph_edges)]

            # Generate Architecture
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            for i in range(num_graph_edges):
                if i not in keep_normal:
                    for j in range(len(switches_normal.current_ops)):
                        switches_normal[i][j] = False
                if i not in keep_reduce:
                    for j in range(len(switches_reduce.current_ops)):
                        switches_reduce[i][j] = False

            switches_normal.keep_2_branches(normal_prob)
            switches_reduce.keep_2_branches(reduce_prob)
            switches_normal.plot_ops(logging.info, writer, cycle.num + 1)
            switches_reduce.plot_ops(logging.info, writer, cycle.num + 1)
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            save_genotype(args.save + 'genotype.json', genotype)
            with open(args.save + "/best_genotype.txt", "w") as f:
                f.write(str(genotype))
    gpu_logger.stop()


def train(train_iterator, valid_iterator, model, criterion, optimizer, optimizer_a, gpu_logger, train_arch=True):
    # trains for one epoch
    logging.debug('training one epoch')
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    inc_batch_size = args.batch_size_max > args.batch_size_min
    for step in train_iterator.yield_steps():

        logging.debug('train step %d', step)
        if train_arch:
            optimizer_a.zero_grad()
            input_search, target_search = valid_iterator.__next__()
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.arch_parameters, args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        input_, target = train_iterator.__next__()
        n = input_.size(0)
        input_ = input_.cuda()
        target = target.cuda(non_blocking=True)
        logits = model(input_)
        loss = criterion(logits, target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.net_parameters, args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)
        if args.test and step >= args.test_batches:
            break

        gpu_logger.wakeup()
        # if there is gpu memory free, try to scale the batch size up
        if inc_batch_size and (step + 1) % 5 == 0:
            usage = gpu_logger.get_highest_recent()
            new_batch_size = math.floor(0.95 * train_iterator.batch_size / usage)
            new_batch_size -= new_batch_size % args.batch_multiples
            new_batch_size = min(new_batch_size, args.batch_size_max, 2*train_iterator.batch_size)
            if new_batch_size > train_iterator.batch_size:
                logging.info('Set new batch size from %d to %d, cur usage is %.2f (step %d)' %
                             (train_iterator.batch_size, new_batch_size, usage, gpu_logger.step_recent))
                train_iterator.set_batch_size(new_batch_size)
                valid_iterator.set_batch_size(new_batch_size)
                torch.cuda.empty_cache()

    return top1.avg, objs.avg


def infer(valid_iterator, model, criterion):
    logging.debug('infering one epoch')
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step in valid_iterator.yield_steps():
        logging.debug('infer step %d', step)
        input_, target = valid_iterator.__next__()
        input_ = input_.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input_)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input_.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if args.test:
            break

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switch):
        n = 2
        start = 0
        gene = []
        for i in range(args.blocks):
            end = start + n
            for j in range(start, end):
                for k in range(len(switch[j])):
                    if switch[j][k]:
                        op_name, op_kwargs = switch.current_ops[k]
                        gene.append((op_name, op_kwargs, j - start))
            start = end
            n = n + 1
        return gene

    concat = list(range(2, args.blocks+2))
    genotype = Genotype(
        normal=_parse_switches(switches_normal),
        normal_concat=concat,
        reduce=_parse_switches(switches_reduce),
        reduce_concat=concat
    )
    return genotype


class SwitchManager:
    """
    utility class to keep track of switches/ops and bundle related methods
    """

    def __init__(self, num_edges, current_ops, name):
        self.current_ops = copy.deepcopy(current_ops)
        self.current_ops_strs = [op_as_str(*op) for op in self.current_ops]
        self.remaining = np.full((num_edges, len(current_ops)), True, dtype=np.bool)    # remaining unless dropped
        self.discovered = np.full((num_edges, len(current_ops)), True, dtype=np.bool)   # discovered by morphing
        self.morphed_from = [{} for _ in range(num_edges)]  # track which op created the new ones
        self.name = name

    def copy(self):
        return copy.deepcopy(self)

    @property
    def switches(self):
        return np.logical_and(self.remaining, self.discovered)

    def __str__(self):
        return str(self.switches)

    def __getitem__(self, y):
        return self.switches[y]

    def __setitem__(self, y, val):
        self.remaining[y] = val

    def __len__(self):
        return len(self.remaining)

    def count_skip_connections(self):
        count, switches = 0, self.switches
        for i, (op_name, _) in enumerate(self.current_ops):
            if 'none' == op_name:
                for j in range(len(switches)):
                    if switches[j][i]:
                        count = count + 1
        return count

    def get_drop_idx(self, probabilities, num_drop, idxs, no_zero=False, keep_morphable=False) -> list:
        """
        figure out which paths to drop for one op,
        optionally guarantee to drop the zero path,
        optionally guarantee to keep at least 1 morphable op (only conv)
        """
        w, idxs = copy.deepcopy(probabilities).tolist(), copy.deepcopy(idxs).tolist()
        zero_idx = []
        if no_zero:
            for i, idx in enumerate(idxs):
                if 'none' == self.current_ops[idx][0]:
                    w.pop(i)
                    idxs.pop(i)
                    zero_idx.append(i)
        rem_idx, _ = list(zip(*sorted(zip(idxs, w), key=lambda v: v[1], reverse=False)))
        rem_idx = zero_idx + list(rem_idx)
        if keep_morphable:
            for i in range(len(rem_idx)-1, -1, -1):
                if 'conv' in self.current_ops[rem_idx[i]][0]:
                    rem_idx.pop(i)
                    break
        return rem_idx[:num_drop]

    def drop_and_morph(self, probabilities, num_keep, writer_, step, num_morphs=0, no_zero=False, keep_morphable=True):
        """
        drop paths per op so that only 'num_keep'-'num_morph' remain,
        optionally guarantee to drop the zero path,
        optionally guarantee to keep at least one morphable op,
        then adds 'num_morph' new morphed operations, if possible
        """
        switches = self.switches
        morphed_with_probabilities = []

        for path in range(len(self.remaining)):
            # list indices of current options
            idxs = np.nonzero(switches[path])[0]
            # drop operations until only num_keep - num_morphs are left
            num_drop = len(idxs) - num_keep + num_morphs
            drop_idx = self.get_drop_idx(probabilities[path, :], num_drop, idxs, no_zero, keep_morphable)
            for idx in drop_idx:
                self.remaining[path][idx] = False
            # insert num_morphs 'new' operations, based on morphed old ones
            if num_morphs > 0:
                morphed_with_p = []
                # remove dropped ops from the list, so we only morph from remaining ops
                idxs = copy.deepcopy(idxs).tolist()
                p = copy.deepcopy(probabilities[path, :]).tolist()
                for idx in drop_idx:
                    i = idxs.index(idx)
                    idxs.pop(i)
                    p.pop(i)
                # idxs sorted by the probability vector, more likely ones are at the front
                morph_idxs, morph_probs = list(zip(*sorted(zip(idxs, p), key=lambda v: v[1], reverse=True)))
                morph_probs = [m / sum(morph_probs) for m in morph_probs]

                # sample an op with respect to their probabilities, morph it until we find something new if possible
                for _ in range(num_morphs):
                    for t in range(250):
                        idx = np.random.choice(morph_idxs, 1, p=morph_probs)[0]
                        # can morph? if not, continue with next sampled op
                        op_name, op_kwargs, as_str = get_morphed_kwargs(morph_restrictions, *self.current_ops[idx])
                        if op_name is None:
                            continue
                        with_prob = morph_probs[morph_idxs.index(idx)]
                        # if the new configuration is yet unknown, add it, keep switch tables correct
                        if as_str not in self.current_ops_strs:
                            self.current_ops.append((op_name, op_kwargs))
                            #import ipdb;ipdb.set_trace()
                            self.current_ops_strs.append(as_str)
                            self.remaining = np.concatenate(
                                [self.remaining, np.full((len(self.remaining), 1), True, dtype=np.bool)], axis=1)
                            self.discovered = np.concatenate(
                                [self.discovered, np.full((len(self.discovered), 1), False, dtype=np.bool)], axis=1)
                        # check if the morphed op is already active or was tested in the past, if not, test it now
                        idx_of_str = self.current_ops_strs.index(as_str)
                        if self.remaining[path][idx_of_str] and not self.discovered[path][idx_of_str]:
                            self.discovered[path][idx_of_str] = True
                            self.morphed_from[path][str(idx_of_str)] = str(idx)  # nn.ModuleDict requires str indexing
                            morphed_with_p.append(with_prob)
                            break
                        # if the op was already discovered and no longer remaining,
                        # but we're already searching for a while, try it out once again
                        if not self.remaining[path][idx_of_str] and self.discovered[path][idx_of_str] and t > 200:
                            self.remaining[path][idx_of_str] = True
                            self.morphed_from[path][str(idx_of_str)] = str(idx)  # this causes cycles :(
                            morphed_with_p.append(with_prob)
                            break
                if len(morphed_with_p) > 0:
                    mean = sum(morphed_with_p)/len(morphed_with_p)
                    writer_.add_scalar('switches_%s_morph_prob/p%d' % (self.name, path), mean, step)
                    morphed_with_probabilities.append(mean)
        if len(morphed_with_probabilities) > 0:
            mean = sum(morphed_with_probabilities)/len(morphed_with_probabilities)
            writer_.add_scalar('switches_%s_morph_prob/avg' % self.name, mean, step)
        return self

    def plot_ops(self, print_fun, writer_, step):
        """
        logs current operations via print_fun,
        writes some stats of op similarity, op types, and convolution params to tensorboard
        """
        switches = self.switches
        sim = []
        avg_op_types = {'conv': 0, 'none': 0, 'pool': 0, 'skip': 0}
        for path in range(len(switches)):
            available_ops = []
            op_types = {'conv': 0, 'none': 0, 'pool': 0, 'skip': 0}
            conv_k = {1: 0, 3: 0, 5: 0, 7: 0}
            for idx in np.nonzero(switches[path])[0]:
                available_ops.append(self.current_ops[idx])
                op_name, op_kwargs = self.current_ops[idx]
                op_types[op_name] = op_types.get(op_name, 0) + 1
                avg_op_types[op_name] = avg_op_types.get(op_name, 0) + 1
                if op_name == 'conv':
                    conv_k[op_kwargs['k']] = conv_k.get(op_kwargs['k'], 0) + 1
            # print available ops
            print_fun('%s, %d: %s' % (self.name, path, str(available_ops)))
            # log path info to tensorboard
            op_sim = utils.op_similarity(available_ops)
            if op_sim is not None:
                sim.append(op_sim)
                writer_.add_scalar('switches_%s_similarity/p%d' % (self.name, path), op_sim, step)
            s = sum(list(op_types.values()))
            s = s if s > 0 else 1
            for k, v in op_types.items():
                writer_.add_scalar('switches_%s_type/p%d/%s' % (self.name, path, k), v / s, step)
            for k, v in conv_k.items():
                writer_.add_scalar('switches_%s_conv_k/p%d/k%s' % (self.name, path, k), v / s, step)
        # log avg info to tensorboard
        writer_.add_scalar('switches_%s_similarity/avg' % self.name, sum(sim) / len(sim), step)
        for k, v in avg_op_types.items():
            writer_.add_scalar('switches_%s_type/avg/%s' % (self.name, k), v/sum(list(avg_op_types.values())), step)

    def keep_2_branches(self, probabilities):
        """ see which branches (node inputs) to keep, switch the others off """

        switches = self.switches
        final_prob = [0.0 for _ in range(len(switches))]
        for i in range(len(switches)):
            final_prob[i] = max(probabilities[i])
        keep = [0, 1]
        n = 3
        start = 2
        for i in range(3):
            end = start + n
            tb = final_prob[start:end]
            edge = sorted(range(n), key=lambda x: tb[x])
            keep.append(edge[-1] + start)
            keep.append(edge[-2] + start)
            start = end
            n = n + 1
        for i in range(len(switches)):
            if i not in keep:
                for j in range(len(self.current_ops)):
                    self.remaining[i][j] = False
        return self


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds.', duration)
