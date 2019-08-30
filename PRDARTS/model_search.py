import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from operations import make_op, FactorizedReduce, ReLUConvBN
from torch.autograd import Variable


class MixedOp(nn.Module):
    def __init__(self, c, stride, switches, index, p):
        super(MixedOp, self).__init__()
        self.m_ops = nn.ModuleDict()
        self.p = p
        switch = switches[index]
        for i in range(len(switch)):
            if switch[i]:
                name, kwargs = switches.current_ops[i]
                ops = [make_op(name, c, stride, False, kwargs)]
                if ops[0].is_pooling_op():
                    ops.append(nn.BatchNorm2d(c, affine=False))
                if ops[0].is_identity_op() and p > 0:
                    ops.append(nn.Dropout(self.p))
                self.m_ops[str(i)] = nn.Sequential(*ops)
                
    def update_p(self, p: float):
        """ find all identity ops and set the dropout probability """
        self.p = p
        for k, sequential in self.m_ops.items():
            if sequential[0].is_identity_op():
                sequential[-1].p = p

    def init_morphed(self, init_from: dict):
        """ initializes morphed ops from their predecessors, if they are still available """
        for k1 in self.m_ops.keys():
            f = init_from.get(k1, -1)    # where the op is originating from
            if f not in self.m_ops:
                continue
            self.m_ops[k1][0].init_from(self.m_ops[f][0])
                    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.m_ops.values()))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, c_prev_prev, c_prev, c, reduction, reduction_prev, switches, p):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.p = p
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self.cell_ops = nn.ModuleList()
        switch_count = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(c, stride, switches=switches, index=switch_count, p=self.p)
                self.cell_ops.append(op)
                switch_count = switch_count + 1
    
    def update_p(self, p: float):
        self.p = p
        for op in self.cell_ops:
            op.update_p(p)

    def init_morphed(self, switches):
        for i in range(len(self.cell_ops)):
            self.cell_ops[i].init_morphed(switches.morphed_from[i])

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self.cell_ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, c, num_classes, layers, criterion, switches_normal, switches_reduce,
                 steps=4, multiplier=4, stem_multiplier=3, p=0.0, largemode=False):
        super(Network, self).__init__()
        logging.debug('building model: %s' % repr(locals()))
        self._c = c
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.p = p
        self.switches_normal = switches_normal
        self.largemode = largemode
        self.num_ops = max(sum(switches_normal[i]) for i in range(len(switches_normal)))

        logging.debug('building stem')
        if self.largemode:
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, c // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 2, c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c),
            )

            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c),
            )
            c_prev_prev, c_prev, c_curr = c, c, c
        else:
            c_curr = stem_multiplier * c
            self.stem = nn.Sequential(
                nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_curr)
            )
    
            c_prev_prev, c_prev, c_curr = c_curr, c_curr, c
        self.cells = nn.ModuleList()
        reduction_prev = self.largemode
        for i in range(layers):
            logging.debug('building cell %d' % i)
            if i in [layers//3, 2*layers//3]:
                c_curr *= 2
                reduction = True
                cell = Cell(steps, multiplier, c_prev_prev, c_prev, c_curr, reduction, reduction_prev, switches_reduce, self.p)
            else:
                reduction = False
                cell = Cell(steps, multiplier, c_prev_prev, c_prev, c_curr, reduction, reduction_prev, switches_normal, self.p)
            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev, multiplier*c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

        # params
        logging.debug('init arc params')
        self._initialize_alphas()
        self.arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        logging.debug('split net/arc params for the optimizers')
        self.net_parameters = []
        for k, v in self.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                self.net_parameters.append(v)
        logging.debug('built model')
        logging.info('Network: %d layers, %d/(%d, %d) ops remaining' %
                     (layers, self.num_ops, len(switches_normal[0]), len(switches_reduce[0])))

    def forward(self, input_):
        if self.largemode:
            s0 = self.stem0(input_)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(input_)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                dim = 0 if self.alphas_reduce.size(1) == 1 else -1
                weights = F.softmax(self.alphas_reduce, dim=dim)
            else:
                dim = 0 if self.alphas_normal.size(1) == 1 else -1
                weights = F.softmax(self.alphas_normal, dim=dim)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def init_morphed(self, switches_normal, switches_reduce):
        """ initializes morphed ops from their predecessors, if they are still available """
        for cell in self.cells:
            cell.init_morphed(switches_reduce if cell.reduction else switches_normal)

    def update_p(self, p):
        self.p = p
        for cell in self.cells:
            cell.update_p(p)
    
    def _loss(self, input_, target):
        logits = self(input_)
        return self._criterion(logits, target)

    def _random_alpha_weight(self):
        # each cell has a matrix of size (num_edges, num_ops)
        k = sum(1 for i in range(self._steps) for _ in range(2+i))
        return 1e-3*torch.randn(k, self.num_ops).cuda()

    def _initialize_alphas(self):
        self.alphas_normal = Variable(self._random_alpha_weight(), requires_grad=True)
        self.alphas_reduce = Variable(self._random_alpha_weight(), requires_grad=True)

    def reset_alphas(self):
        self.alphas_normal.data.zero_().add_(self._random_alpha_weight())
        self.alphas_reduce.data.zero_().add_(self._random_alpha_weight())

    def randomize_alphas(self):
        # just for debugging/testing
        self.alphas_normal.data.add_(torch.rand_like(self.alphas_normal.data)*10 - 5)
        self.alphas_reduce.data.add_(torch.rand_like(self.alphas_reduce.data)*10 - 5)
