import torch
import torch.nn as nn
import torch.nn.functional as F

PRIMITIVES = [
    'none',
    'skip_connect',
    'cweight_com',
    'avg_pool_3x3',
    'max_pool_3x3',
    'sep_conv_3x3',
    'dil_conv_3x3',
    'shuffle_conv_3x3',
]

# Original Darts
# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5'
# ]

#TODO(zbaby: 2018/9/28) Deform Convolution
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'cweight_com': lambda C, stride, affine: CWeightCom(C, C, 3, stride, 1, affine=affine),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'shuffle_conv_3x3': lambda C, stride, affine: ShuffleConv(C, C, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):
    """ReLU -> Convolution Layer -> BatchNorm Layer

    When we say add a convolution layer, it actually add
    ReLU activation function, Convolution Layer and BatchNorm Layer

    Attributes:
        c_in: int, the input channels to convolution layer
        c_out: int, the output channel by convolution layer
        kernel_size: int, the kernel size of filters
        stride: int, the stride of convolution layer
        padding: int, convolution layer
        affine: boolean if we need affine for batch norm,default: True
    """
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out, affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Dilation convolution with specified order

    A specified order layer with dilation convolution:
    ReLU -> DilConv -> Conv2d -> BatchNorm2d

    Attributes:
        c_in: A integer representing the input channels to Dilation convolution layer
        c_out: A integer representing the output channel by Dilation convolution layer
        kernel_size: A integer representing the kernel size of filters
        stride: A integer representing the stride of convolution layer
        padding: A integer for Dilation convolution layer
        affine: A boolean indicating if we need affine for batch norm,default: True
    """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            # separate dilation convolution
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=c_in, bias=False),
            # we need a conv1d to concatenate the separate convolution
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Separate convolution

    Separate convolution with specified order:
    ReLU -> Conv2d -> Conv2d -> BatchNorm2d -> ReLU -> Conv2d - > Conv2d -> BatchNorm2d

    Attributes:
        c_in: A integer representing the input channels to Separate convolution layer
        c_out: A integer representing the output channel by Separate convolution layer
        kernel_size: A integer representing the kernel size of filters
        stride: A integer representing the stride of Separate convolution layer
        padding: A integer for Separate convolution layer
        affine: A boolean indicating if we need affine for batch norm,default: True
    """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            # separate convolution
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=c_in, bias=False),
            # we need a conv1d to concatenate the separate convolution
            nn.Conv2d(c_in, c_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=1, padding=padding,
                      groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """Identity layer for combining a complicated cell
    """

    def __init___(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    """Zero operation

    Zero operation for clear up the feature map

    Attributes:
        stride: A integer representing the stride of layer
    """

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.) # N * C * W * H

class FactorizedReduce(nn.Module):
    """Factorized reduction layer to reduce the input dimension

    When the prev cell/blocks is reduction, we need a compatible prev_prev cell input
    so the twice reductive convolution

    Attributes:
        c_in: A integer representing the input channels to Factorized Reduction convolution layer
        c_out: A integer representing the output channels to Factorized Reduction convolution layer
        affine: A boolean indicating if we need affine for batch norm,default: True

    """

    def __init__(self, c_in, c_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert c_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(c_in, c_out//2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out//2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        #Todo(zbaby: 2018/9/28) this implement was odd!
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)

        return out

class CWeightCom(nn.Module):
    """ Channel Weighted Combination For Feature Map

    In the standard convolution layer, we combination the channel use the same weight.
    However, maybe in some feature map, the channel weights are not equal for some reasons.
    Since we are searching work, all of the possible way architecture stacked is consider!

    Attribute:
        c_in: int, the input channels of original feature map
        c_out: int, the output channels of original feature map
        kernel_size: int, if the stride == 2 , we need a 3
    """

    def __init__(self, c_in, c_out, kernel_size, stride, padding=1, affine=True, reduction=16):
        super(CWeightCom, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_out),
            nn.Sigmoid()
        )
        self.stride = stride
        if stride == 2:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                                  padding=padding, groups = c_in, bias=False)
            self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        rst = self.bn(self.conv(x*y)) if self.stride == 2 else x*y

        return rst


class Shuffle(nn.Module):
    """Shuffle The Feature Map"""

    def __init__(self, groups):
        super(Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N, C, H, W] -> [N, g, C/g, g, H, w] - >[N, C, H ,W]"""
        N, C, H, W = x.size()
        assert C % self.groups == 0
        return x.view(N,self.groups, C//self.groups, H, W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class ShuffleConv(nn.Module):
    """Shuffle Convolution

    Attributes:
        c_in: int, input data/feature map channels
        c_out: int, output feature map channels
        stride: int, step
        groups: int,
    """

    def __init__(self, c_in, c_out, stride, groups=4, padding=1, affine=True):
        super(ShuffleConv, self).__init__()
        self.stride = stride
        c_mid = c_out // groups
        self.relu = nn.ReLU(inplace=False)
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(c_mid, affine=affine),
            nn.ReLU(inplace=True),
            Shuffle(groups=groups),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride,
                      padding=padding, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid, affine=affine),
            nn.Conv2d(c_mid, c_out, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=stride, padding=padding))

    def forward(self, x):
        out1 = self.op(x)
        out2 = self.shortcut(x)
        out = F.relu(out1+out2)
        return out










