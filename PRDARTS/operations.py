import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphRestrictions:
    def __init__(self, conv_min_k=1, conv_max_k=7, conv_max_dil=2,
                 conv_max_c_mul_len=2, conv_max_c_mul_num=3, conv_max_c_mul_width=3):
        self.conv_min_k = conv_min_k
        self.conv_max_k = conv_max_k
        self.conv_max_dil = conv_max_dil
        self.conv_max_c_mul_len = conv_max_c_mul_len
        self.conv_max_c_mul_num = conv_max_c_mul_num
        self.conv_max_c_mul_width = conv_max_c_mul_width


def expand_tensor(tensor, c, dim, init_fun=None, mult=0):
    size = list(tensor.size())
    size[dim] = c
    w = torch.ones(size, dtype=tensor.dtype) * mult
    if init_fun is not None:
        init_fun(w)
    return torch.cat([tensor.clone().detach(), w], dim=dim).requires_grad_(tensor.requires_grad)


class Conv2d(nn.Conv2d):
    """ like nn.Conv2d, with convenience methods """

    def make_identity(self):
        """ sets the weight of this conv so that it resembles an identity op """
        k1, k2 = self.kernel_size
        if self.groups > 1:
            self.weight.data.zero_()[:, :, k1//2, k2//2] = 1
        else:
            nn.init.dirac_(self.weight.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def expand_channels(self, c, dim, init_fun=None):
        """ expands the weights of this conv along dim by 'c', new weights are initialized in place with 'init_fun' """
        if dim == 0:
            self.out_channels += c
        else:
            self.in_channels += c
        if dim == 0 and self.groups > 1:
            self.groups += c
            self.in_channels += c
        if dim == 0 and self.bias is not None:
            self.bias.data = expand_tensor(self.bias, c, 0, init_fun)
        self.weight.data = expand_tensor(self.weight, c, dim, init_fun)

    def init_from(self, other):
        """
        initializes params as much as possible according to 'other'
        """
        k_d, k = (self.kernel_size[0] - other.kernel_size[0]) // 2, other.kernel_size[0]
        ci_d, ci = self.in_channels - other.in_channels, other.in_channels
        co_d, co = self.out_channels - other.out_channels, other.out_channels
        w_tensor = other.weight.data.clone() if k_d >= 0 else self.weight.data.clone()
        if ci_d > 0:
            w_tensor = expand_tensor(other.weight, ci_d, dim=1, init_fun=None)
        if ci_d < 0:
            w_tensor = w_tensor[:, :-ci_d, :, :]
        if co_d > 0:
            w_tensor = expand_tensor(other.weight, co_d, dim=0, init_fun=nn.init.xavier_normal_)
        if co_d < 0:
            w_tensor = w_tensor[:-co_d, :, :, :]
        if k_d > 0:
            w_tensor = F.pad(w_tensor, [k_d, k_d, k_d, k_d])
            w_tensor[:co, :ci, k_d:-k_d, k_d:-k_d] = other.weight.data.clone()
        if k_d < 0:
            k_d = abs(k_d)
            w_tensor[:co, :ci, :, :] = other.weight.data.clone()[:, :, k_d:-k_d, k_d:-k_d]
        self.weight.data = w_tensor
        self.weight.requires_grad_(other.weight.requires_grad)
        if self.bias is not None and other.bias is not None:
            if co_d > 0:
                self.bias.data = expand_tensor(other.bias, co_d, dim=0)
            if co_d < 0:
                self.bias.data = other.bias[:-co_d]


class BatchNorm2d(nn.BatchNorm2d):
    """ like nn.BatchNorm2d, with convenience methods """

    def make_identity(self):
        self.reset_parameters()

    def expand_channels(self, c, other=None):
        other = self if other is None else other
        if self.affine:
            self.weight.data = expand_tensor(other.weight, c, 0, None)
            self.bias.data = expand_tensor(other.bias, c, 0, None)
        self.running_mean.data = expand_tensor(other.running_mean, c, 0, None)
        self.running_var.data = expand_tensor(other.running_var, c, 0, None, mult=1)

    def init_from(self, other):
        if self.affine:
            c_d = self.weight.data.size(0) - other.weight.data.size(0)
        else:
            c_d = self.running_mean.data.size(0) - other.running_mean.data.size(0)
        return self.expand_channels(c_d, other=other)


class SepConv(nn.Module):
    def __init__(self, c_in, c_out, stride, affine, k, dil, bias=False):
        super(SepConv, self).__init__()
        p = ((k-1) * dil) // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            Conv2d(c_in, c_in, kernel_size=k, stride=stride, padding=p, dilation=dil, groups=c_in, bias=bias),
            Conv2d(c_in, c_out, kernel_size=1, padding=0, bias=bias),
            BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

    def init_from(self, other):
        """ init weights to be as similar to 'other' as possible """
        self.op[-3].init_from(other.op[-3])   # point wise conv
        self.op[-2].init_from(other.op[-2])   # depth wise conv
        self.op[-1].init_from(other.op[-1])   # batchnorm

    def make_identity(self):
        """ sets the weight of this conv so that it resembles an identity op """
        self.op[-3].make_identity()     # point wise conv
        self.op[-2].make_identity()     # depth wise conv
        self.op[-1].make_identity()     # batchnorm


class Op(nn.Module):
    def __init__(self, c_in, c_out, stride, affine, unused_kwargs, **kwargs):
        super(Op, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.affine = affine
        self.kwargs = kwargs
        if len(unused_kwargs) > 0:
            print('unused kwargs!', repr(self), unused_kwargs)

    @staticmethod
    def label_str(**kwargs):
        raise NotImplementedError

    @staticmethod
    def get_morphed_kwargs(restrictions: MorphRestrictions, **kwargs):
        return None

    def init_from(self, other):
        """ init weights to be as similar to 'other' as possible """
        pass

    def is_identity_op(self):
        # identity ops may have additional dropout during search
        return False

    def is_pooling_op(self):
        # pooling ops may have additional batchnorm during search
        return False


# NOTE consider morphing k!
class PoolOp(Op):
    def __init__(self, c_in, c_out, stride, affine, k=3, type_='avg', **_):
        super(PoolOp, self).__init__(c_in, c_out, stride, affine, _, k=k, type_=type_)
        op, kwargs = {
            'avg': (nn.AvgPool2d, {'count_include_pad': False}),
            'max': (nn.MaxPool2d, {}),
        }.get(type_)
        self.op = op(k, stride=stride, padding=k // 2, **kwargs)

    @staticmethod
    def label_str(k=3, type_='avg'):
        return 'Pool(%s, %dx%d)' % (type_, k, k)

    def is_pooling_op(self):
        return True

    def forward(self, x):
        return self.op(x)


class ConvOp(Op):
    def __init__(self, c_in, c_out, stride, affine, c_mul=[], k=3, dil=1, **_):
        super(ConvOp, self).__init__(c_in, c_out, stride, affine, _, c_mul=list(c_mul), k=k, dil=dil)
        ops, cur_c_in, cur_c_out = [], c_in, c_in
        for i, mult in enumerate(c_mul):
            cur_c_out = int(c_in * mult)
            ops.append(SepConv(cur_c_in, cur_c_out, stride if i == 0 else 1, affine, k=k, dil=dil))
            cur_c_in = cur_c_out
        ops.append(SepConv(cur_c_out, c_out, stride if len(ops) == 0 else 1, affine, k=k, dil=dil))
        self.op = nn.Sequential(*ops)

    @staticmethod
    def label_str(c_mul=[], k=3, dil=1):
        return 'Conv({k}x{k}{dil}{mult})'.format(**{
            'mult': '' if len(c_mul) == 0 else ', mult=%s' % str(c_mul),
            'k': k,
            'dil': '' if dil == 1 else ', dil=%d' % dil
        })

    def init_from(self, other):
        """ init weights to be as similar to 'other' as possible """
        for i in range(0, len(other.op)):
            self.op[i].init_from(other.op[i])
        for i in range(len(other.op), len(self.op)):
            self.op[i].make_identity()

    @staticmethod
    def get_morphed_kwargs(restrictions: MorphRestrictions, c_mul=[], k=3, dil=1):
        c_mul = copy.deepcopy(c_mul)
        while True:
            r = np.random.randint(0, 6, 1)[0]
            if r == 0:
                # increase k if possible
                if k+2 > restrictions.conv_max_k:
                    continue
                k += 2
            elif r == 1:
                # decrease k if possible
                if k-2 < restrictions.conv_min_k:
                    continue
                if k-2 == 1 and dil >= 2:
                    continue
                k -= 2
            elif r == 2:
                # increase dilation
                if k == 1 or dil >= restrictions.conv_max_dil:
                    continue
                dil += 1
            elif r == 3:
                # decrease dilation
                if dil <= 1 or (k == 1 and dil == 2):
                    continue
                dil -= 1
            elif r == 4:
                # insert convolution
                if len(c_mul) >= restrictions.conv_max_c_mul_len or sum(c_mul) >= restrictions.conv_max_c_mul_num:
                    continue
                c_mul.append(1)
            elif r == 5:
                # expand the smallest convolution if available
                if len(c_mul) <= 0:
                    continue
                idx = np.argmin(c_mul).__int__()
                if c_mul[idx] >= restrictions.conv_max_c_mul_width or sum(c_mul) >= restrictions.conv_max_c_mul_num:
                    continue
                c_mul[idx] += 1
            return {
                'c_mul': c_mul,
                'k': k,
                'dil': dil,
            }

    def forward(self, x):
        return self.op(x)


class SkipOp(Op):
    def __init__(self, c_in, c_out, stride, affine, **_):
        super(SkipOp, self).__init__(c_in, c_out, stride, affine, _)
        self.stride = stride
        if stride == 1:
            self.op = lambda x: x
        else:
            self.op = FactorizedReduce(c_in, c_out, affine=affine)

    @staticmethod
    def label_str():
        return 'Skip()'

    def is_identity_op(self):
        return self.stride == 1

    def forward(self, x):
        return self.op(x)


class ZeroOp(Op):
    def __init__(self, c_in, c_out, stride, affine, **_):
        super(ZeroOp, self).__init__(c_in, c_out, stride, affine, _)
        self.stride = stride

    @staticmethod
    def label_str():
        return 'Zero()'

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
        else:
            padding = torch.FloatTensor(n, c, h, w).fill_(0)
        return padding


# just for AmoebaNet
class Conv7x1x1x7Op(Op):
    def __init__(self, c_in, c_out, stride, affine, **_):
        super(Conv7x1x1x7Op, self).__init__(c_in, c_out, stride, affine, _)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_out, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(c_out, c_out, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

    @staticmethod
    def label_str():
        return 'Conv(1x7, 7x1)'

    def forward(self, x):
        return self.op(x)


# used in model.py to correct channel sizes
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


# for SkipOp and more
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


OPS = {
    'none': ZeroOp,
    'pool': PoolOp,
    'skip': SkipOp,
    'conv': ConvOp,
    'conv_7x1_1x7': Conv7x1x1x7Op
}


def make_op(name, c, stride, affine, kwargs) -> Op:
    return OPS.get(name)(c, c, stride, affine, **kwargs)


def op_as_str(name, kwargs) -> str:
    values = sorted(zip(kwargs.keys(), kwargs.values()), key=lambda v: v[0])
    return name+str(values)


def get_morphed_kwargs(restrictions: MorphRestrictions, name, kwargs) -> (str, dict, str):
    morphed_kwargs = OPS.get(name).get_morphed_kwargs(restrictions, **kwargs)
    if morphed_kwargs is None:
        return None, None, None
    return name, morphed_kwargs, op_as_str(name, morphed_kwargs)


def get_morph_restrictions(type_: str) -> MorphRestrictions:
    """
    returns a MorphRestrictions object, limiting morphisms to desired search spaces
    'darts_like'        almost like DARTS, may end up with stacked dilated sep convs or unstacked sep convs however
    'unrestricted'      no restrictions imposed on expansion ratios
    'r_depth'           expansion ratios may be wide but not stacked
    """
    return {
        'darts_like': MorphRestrictions(conv_min_k=3, conv_max_k=7, conv_max_dil=2,
                                        conv_max_c_mul_len=1, conv_max_c_mul_num=1, conv_max_c_mul_width=1),
        'unrestricted': MorphRestrictions(conv_min_k=1, conv_max_k=7, conv_max_dil=2,
                                          conv_max_c_mul_len=999, conv_max_c_mul_num=999, conv_max_c_mul_width=999),
        'r_depth': MorphRestrictions(conv_min_k=1, conv_max_k=7, conv_max_dil=2,
                                                conv_max_c_mul_len=1, conv_max_c_mul_num=5, conv_max_c_mul_width=5),
    }.get(type_.lower())
