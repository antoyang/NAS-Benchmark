import torch
import torch.nn as nn
from darts.operations import *
from darts.utils import drop_path

class BuildCell(nn.Module):
    """Build a cell from genotype

    Genotype represent a coding of architecture for cell, includes, topology of feature map,
    operation(like separate convolution, skip connect..) apply for feature map

    Attributes:
        genotype: tuple, coding of architecture
        c_prev_prev: int, prev prev cell output channels
        c_prev: int, prev cell output channels
        c: int, current cell output channels
        reduction: boolean, whether current cell is reduction cell
        reduction_prev: boolean, whether prev cell is reduction cell
    """

    def __init__(self, genotype, c_prev_prev, c_prev, c, reduction, reduction_prev):
        super(BuildCell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c, 1, 1, 0)

        if reduction:
            op_names, idx = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, idx = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(c, op_names, idx, concat, reduction)
        # self._remove_zero_op(c, op_names, idx, concat, reduction)

    def _compile(self, c, op_names, idx, concat, reduction):
        assert len(op_names) == len(idx)
        # the input number of meta node always two
        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, idx):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](c, stride, True)
            self._ops += [op]
        self._indices = idx

        # TODO(zbabby(2018/10/9) delete the zero node (op1 and op2 are none operation)

    def _remove_zero_op(self, c, op_names, idx, concat, reduction):

        self._num_meta_node = len(op_names) // 2
        self._concat = concat
        self.multiplier = 0
        # Pytorch can't support a null list, so need a auxiliary python list
        self._ops = nn.ModuleList()
        self.res_list = [[] for i in range(self._num_meta_node+2)]
        self.res_list[0].append('prev')
        self.res_list[1].append('prev_prev')
        for i in range(self._num_meta_node):

            # Get the operation name applied to prev feature map
            op1 = op_names[2*i]
            idx1 = idx[2*i]
            op2 = op_names[2*i+1]
            idx2 = idx[2*i+1]

            valid = False
            if op1 != 'none' and len(self.res_list[idx1]) != 0:
                stride = 2 if reduction and idx1 in [0, 1] else 1
                self.res_list[i+2].append(idx1)
                self._ops += [OPS[op1](c, stride, True)]
                valid = True
            if op2 != 'none' and len(self.res_list[idx2]) != 0:
                stride = 2 if reduction and idx2 in [0, 1] else 1
                self.res_list[i+2].append(idx2)
                self._ops += [OPS[op2](c, stride, True)]
                valid = True

            if valid:
                self.multiplier += 1

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._num_meta_node):
            # Get the input node
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            # Get the operation applied to prev feature map(node)
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]

            h1 = op1(h1)
            h2 = op2(h2)
            if self.training:
                # if not isinstance(op1, Identity):
                #     h1 = drop_path(h1, drop_prob)
                # if not isinstance(op2, Identity):
                #     h2 = drop_path(h2, drop_prob)
                if not isinstance(op1, Identity) and not isinstance(op1,Zero):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity) and not isinstance(op2,Zero):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class AuxiliaryHead(nn.Module):
    """Auxiliary classifier layer for cifar10, imagenet and etc

    Auxiliary classifier layer for cifar10, cifar100 and etc, which
    input image size assuming 8x8( an auxiliary tower which backpropagates
    the classification loss earlier in the network,
    serving as an additional regularization mechanism. For
    simplicity,
    ref: 1. Xception: Deep Learning with Depthwise Separable Convolutions
         2. Going deeper with convolutions
    )

    Attributes:
        c: int, initial channel for auxiliary classifier layer
        num_classes: int, 10/100/101/200/1000 etc.
    """

    def __init__(self, c, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            # after avgpool image size is 2x2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(c, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class EvalNetwork(nn.Module):
    """Construct a network for ImageNet

    Attributes:
        c: int, the initial channels of network
        num_classes: int, the number of object to classified
        layers: int, the total number of cells
        auxiliary: boolean, whether use a auxiliary classifier layer
        genotype: tuple, coding of architecture
        reduce_level: int, the level of reduce dimension of input images before put into network
                      0, 1, 2: [0,32], [32,64], [64, 128]
    """

    def __init__(self, c, num_classes, drop_path_prob, layers, auxiliary, genotype, reduce_level=0):
        super(EvalNetwork, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._drop_path_prob = drop_path_prob
        self._reduce_level = reduce_level

        stem_multiplier = 3
        c_curr = stem_multiplier * c
        if self._reduce_level == 0:
            self.stem = nn.Sequential(
                nn.Conv2d(3, c_curr, kernel_size=3,padding=1, bias=False),
                nn.BatchNorm2d(c_curr)
            )
        elif self._reduce_level == 1:
            self.stem = nn.Sequential(
              nn.Conv2d(3, c_curr, 3, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(c_curr),
            )
        elif self._reduce_level == 2:
            self.stem = nn.Sequential(
                nn.Conv2d(3, c_curr // 2, kernel_size=3,padding=1, bias=False),
                nn.BatchNorm2d(c_curr // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_curr // 2, c_curr, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_curr // 2)
            )
        elif self._reduce_level == 3:
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, c_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_curr // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_curr // 2, c_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_curr),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(c_curr, c_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_curr),
            )

        if reduce_level == 3:
            reduction_prev = True
        else:
            reduction_prev = False
        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c
        self.cells = nn.ModuleList()
        for i in range(0, layers):
            if i in [layers //3, 2*layers//3]:
                c_curr *=2
                reduction = True
            else:
                reduction = False
            cell = BuildCell(genotype, c_prev_prev, c_prev, c_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev, cell.multiplier*c_curr
            if i == 2*layers // 3:
                c_to_auxiliary = c_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(c_to_auxiliary, num_classes)
        if num_classes > 67:
            gp_outsize = 2
        else:
            gp_outsize = 1
        self.global_pooling = nn.AdaptiveAvgPool2d(gp_outsize)

        self.classifier = nn.Linear(c_prev*gp_outsize*gp_outsize, num_classes)

    def forward(self, input):
        logits_aux = None
        if self._reduce_level == 3:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(input)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self._drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits, logits_aux


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy to smooth label

    Attributes:
        num_classes: int, the number of classes
        epsilon: float, the parameter for label smooth
    """
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



