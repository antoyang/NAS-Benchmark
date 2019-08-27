import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.operations import *
from darts.geno_types import *

class MixedOp(nn.Module):
    """Mixed Operations

    Mixed operation using sum of weighted operations
    Attributes:
        c: int, the input channel for operations
        stride: int, the sliding filter step
    """

    def __init__(self, c, stride, affine):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](c, stride, affine=affine)
            if 'pool' in primitive:
                # BatchNorm should make sure affine = False to avoid
                # rescale the output of the candidate operations
                op = nn.Sequential(op, nn.BatchNorm2d(c, affine=affine))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    """The basic cell for stacking our finally architecture

    Cell represent a basic block we searched

    Attributes:
        meta_node_num: int, the number of meta-node in a cell
        multiplier: int, the multiplier
        c_prev_prev: int, the channels of prev prev input cell
        c_prev: int, the channels of prev input cell
        c: int, the channels of current first input node
        reduction: boolean, whether current's cell is a reduction cell
        reduction_prev: boolean, whether prev cell is a reduction
    """

    def __init__(self, meta_node_num, c_prev_prev, c_prev, c, reduction, reduction_prev, affine=False):
        super(Cell, self).__init__()
        self.reduction = reduction

        #
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c, affine=affine)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c, 1, 1, 0, affine=affine)

        self.preprocess1 =ReLUConvBN(c_prev, c, 1, 1, 0, affine=affine)
        self._metan_num = meta_node_num
        self._multiplier = meta_node_num
        self._inp_node_num = 2

        self._ops = nn.ModuleList()

        for i in range(self._metan_num): # meta-node id
            for j in range(self._inp_node_num + i): # the input id for remaining meta-node
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(c, stride, affine)
                self._ops.append(op)

    def forward(self, s0, s1, weight):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self._metan_num):
            s = sum(self._ops[offset+j](h, weight[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # Concatenate all meta-node to output along channels dimension
        # Todo(zbabby: 2018/10/6) the later, we will combine more flexibly
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    """A Network Stacked By Cells

    The network is a fully convolution network and not connect any detect layer or fc layer
    Todo(zbabby(2018/9/25))  The network should comparable for all specified vision task
    Attributes:
        inp_c: int, input image channel
        c: int, output channel for input image by first layer
        num_classes: int, number of classes for eval
        layers: int, total number of layers
        criterion: A torch loss, selected loss function
        num_meta_node: int, the number of intermediate node
        stem_multiplier: int, the multi to enlarge the first node output channels
        reduce_level: int, the level of reduce dimension of input images before put into network
                      0, 1, 2: [0,32], [32,64], [64, 128]
        use_sparse: boolean, whether use a sparse architecture
    """

    def __init__(self, inp_c, c, num_classes, layers, criterion, num_inp_node,
                 num_meta_node=4, stem_multiplier=3, reduce_level = 0, use_sparse=False):
        super(Network, self).__init__()
        self._inp_c = inp_c # 3
        self._c = c         # 16
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._num_inp_node = num_inp_node
        self._num_meta_node = num_meta_node
        self._multiplier = num_meta_node
        self._reduce_level = reduce_level
        self._use_sparse = use_sparse

        c_curr = stem_multiplier * c

        if self._reduce_level == 0:
            self.stem = nn.Sequential(
                nn.Conv2d(inp_c, c_curr, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_curr, affine=False if not use_sparse else True)
            )
        elif self._reduce_level == 1:
            self.stem = nn.Sequential(
              nn.Conv2d(inp_c, c_curr, 3, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(c_curr, affine=False if not use_sparse else True)
            )
        elif self._reduce_level == 2:
            self.stem = nn.Sequential(
                nn.Conv2d(inp_c, c_curr // 2, kernel_size=3,padding=1, bias=False),
                nn.BatchNorm2d(c_curr // 2, affine=False if not use_sparse else True),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_curr // 2, c_curr, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_curr, affine=False if not use_sparse else True)
            )
        elif self._reduce_level == 3:
            self.stem0 = nn.Sequential(
                nn.Conv2d(inp_c, c_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
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

        # Construct init architecture by stacking cells
        if reduce_level == 3:
            reduction_prev = True
        else:
            reduction_prev = False
        c_prev_prev, c_prev, c_curr, = c_curr, c_curr, c
        self.cells = nn.ModuleList()
        for i in range(layers):
            #Todo(zbabby:2018/9/27) the position of reduction cell need to do some experiment
            if i in [layers//3, 2*layers//3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(num_meta_node, c_prev_prev, c_prev, c_curr, reduction,
                        reduction_prev, affine=False if not use_sparse else True)
            reduction_prev = reduction
            self.cells += [cell]
            c_prev_prev, c_prev = c_prev,  self._multiplier*c_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(c_prev, num_classes)

        # Initialize architecture parameters: alpha
        self._initialize_alphas()

    def _initialize_alphas(self):
        """Initialize The Architecture Parameter Alpha

        Initialize the architecture parameter \alpha, represented by a
        N * M matrix, which N is the total number of input node, M is the number of operations for input node
        """
        k = sum(1 for i in range(self._num_meta_node) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        # Requires gradient
        self.alphas_normal = torch.tensor(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = torch.tensor(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce
        ]

    def load_alphas(self, alphas_dict):
        """ Load alphas from checkpoint"""
        self.alphas_normal = alphas_dict['alpha_normal']
        self.alphas_reduce = alphas_dict['alpha_reduce']
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce
        ]


    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """Generate Geno Type"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._num_meta_node):
                end = start + n
                W = weights[start:end].copy()

                # Get the k largest strength of mixed op edges, which k = 2
                if self._use_sparse:
                    edges = sorted(range(i+2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                else:
                    edges = sorted(range(i+2), key=lambda x: -max(W[x][k]
                            for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]

                # Get the best operation
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if not self._use_sparse and k == PRIMITIVES.index('none'):
                            continue
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    # Geno item: (operation, node idx)
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        #Todo(zbabby parallize)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2, self._num_meta_node+2)
        geno_type = Genotype(
            normal=gene_normal, normal_concat = concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return geno_type

    def new(self):
        """New A Network: Create a new network by deep copy from old network"""
        #Todo(parallize)
        model_new = Network(self._inp_c, self._c, self._num_classes, self._layers,
                            self._criterion, self._num_inp_node, self._num_meta_node).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        if self._reduce_level == 3:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            # Sharing a global N*M weights matrix
            if cell.reduction:
                # put the alphas into softmax function
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights) # current do forward
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits