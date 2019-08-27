import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

INPLACE=False
BIAS=False


def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
        
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
    
    def forward(self, x, bn_train=False):
        #import  ipdb; ipdb.set_trace()
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=INPLACE)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, bn_train=False):
        x = self.relu(x)
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x, bn_train=False):
        x = self.ops(x)
        return x


class WSReLUConvBN(nn.Module):
    def __init__(self, num_possible_inputs, C_out, C_in, kernel_size, stride=1, padding=0):
        super(WSReLUConvBN, self).__init__()
        self.stride = stride
        self.padding = padding
        self.relu = nn.ReLU(inplace=INPLACE)
        self.w = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, kernel_size, kernel_size)) for _ in range(num_possible_inputs)])
        self.bn = nn.BatchNorm2d(C_out, affine=True)
    
    def forward(self, x, x_id, bn_train=False):
        x = self.relu(x)
        w = torch.cat([self.w[i] for i in x_id], dim=1)
        x = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class WSBN(nn.Module):

    def __init__(self, num_possible_inputs, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(WSBN, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            for i in range(self.num_possible_inputs):
                self.weight[i].data.fill_(1)
                self.bias[i].data.zero_()

    def forward(self, x, x_id, bn_train=False):
        training = self.training
        if bn_train:
            training = True
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight[x_id], self.bias[x_id],
            training, self.momentum, self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__))
    
    
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)


class WSSepConv(nn.Module):
    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(WSSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding
        
        self.relu1 = nn.ReLU(inplace=INPLACE)
        self.W1_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W1_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn1 = WSBN(num_possible_inputs, C_in, affine=affine)

        self.relu2 = nn.ReLU(inplace=INPLACE)
        self.W2_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W2_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn2 = WSBN(num_possible_inputs, C_in, affine=affine)
    
    def forward(self, x, x_id, stride=1, bn_train=False):
        x = self.relu1(x)
        x = F.conv2d(x, self.W1_depthwise[x_id], stride=stride, padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W1_pointwise[x_id], padding=0)
        x = self.bn1(x, x_id, bn_train=bn_train)

        x = self.relu2(x)
        x = F.conv2d(x, self.W2_depthwise[x_id], padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W2_pointwise[x_id], padding=0)
        x = self.bn2(x, x_id, bn_train=bn_train)
        return x


class WSAvgPool2d(nn.Module):
    def __init__(self, kernel_size, padding):
        super(WSAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
    
    def forward(self, x, stride):
        return F.avg_pool2d(x, self.kernel_size, stride, self.padding, count_include_pad=False)


class WSMaxPool2d(nn.Module):
    def __init__(self, kernel_size, padding):
        super(WSMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
    
    def forward(self, x, stride):
        return F.max_pool2d(x, self.kernel_size, stride, self.padding)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        self.multi_adds = 0
        hw = [layer[0] for layer in layers]
        #print(hw)
        c = [layer[-1] for layer in layers]
        
        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        # previous reduction cell
        if hw[0] != hw[1] :
            assert hw[0] == 2 * hw[1]
            self.relu = nn.ReLU(inplace=INPLACE)
            self.preprocess_x = FactorizedReduce(c[0], channels, affine)
            x_out_shape = [hw[1], hw[1], channels]
            #print("x_out_shape:", x_out_shape)
            self.multi_adds += 1 * 1 * c[0] * channels * hw[1] * hw[1]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, affine)
            x_out_shape = [hw[0], hw[0], channels]
            #print("x_out_shape:", x_out_shape)
            self.multi_adds += 1 * 1 * c[0] * channels * hw[1] * hw[1]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, affine)
            y_out_shape = [hw[1], hw[1], channels]
            #print("y_out_shape:", y_out_shape)
            self.multi_adds += 1 * 1 * c[1] * channels * hw[1] * hw[1]
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1, bn_train=False):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1, bn_train=bn_train)
        return [s0, s1]


class FinalCombine(nn.Module):
    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        self.multi_adds = 0
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                #print("hw :", hw)
                #print("out hw :", out_hw)
                assert hw == 2 * out_hw and i in [0,1]
                self.concat_fac_op_dict[i] = len(self.ops)
                self.ops.append(FactorizedReduce(layers[i][-1], channels, affine))
                self.multi_adds += 1 * 1 * layers[i][-1] * channels * out_hw * out_hw
        
    def forward(self, states, bn_train=False):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i], bn_train)
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out


OPERATIONS = {
    0: SepConv, # 3x3
    1: SepConv, # 5x5
    2: nn.AvgPool2d, # 3x3
    3: nn.MaxPool2d, # 3x3
    4: Identity,
}
OPERATIONS_large = {
    5: Identity,
    6: Conv, # 1x1
    7: Conv, # 3x3
    8: Conv, # 1x3 + 3x1
    9: Conv, # 1x7 + 7x1
    10: nn.MaxPool2d, # 2x2
    11: nn.MaxPool2d, # 3x3
    12: nn.MaxPool2d, # 5x5
    13: nn.AvgPool2d, # 2x2
    14: nn.AvgPool2d, # 3x3
    15: nn.AvgPool2d, # 5x5
}