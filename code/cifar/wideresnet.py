# Network code from https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from nested_dict import nested_dict


def wideresnet():
    return WideResNet(28, 2, num_classes=10)


class WideResNet(nn.Module):

    def __init__(self, depth, width, num_classes):
        super(WideResNet, self).__init__()
        f, params = resnet(depth, width, num_classes)
        self.f = f
        for name, param in params.items():
            if name.endswith('running_mean') or name.endswith('running_var'):
                self.register_buffer(name, param)
            else:
                self.register_parameter(name, nn.Parameter(param))

    def forward(self, x):
        params = OrderedDict(self.named_parameters())
        params.update(self.named_buffers())
        return self.f(x, params, self.training)


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': conv_params(ni, no, 3),
            'conv1': conv_params(no, no, 3),
            'bn0': bnparams(ni),
            'bn1': bnparams(no),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = flatten({
        'conv0': conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': bnparams(widths[2]),
        'fc': linear_params(widths[2], num_classes),
    })

    set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(batch_norm(x, params, base + '-bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '-conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, base + '-bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '-conv1'], stride=1, padding=1)
        if base + '-convdim' in params:
            return z + F.conv2d(o1, params[base + '-convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, '%s-block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, 'group0', mode, 1)
        g1 = group(g0, params, 'group1', mode, 2)
        g2 = group(g1, params, 'group2', mode, 2)
        o = F.relu(batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc-weight'], params['fc-bias'])
        return o

    return f, flat_params


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def flatten(params):
    return {'-'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '-weight'],
                        bias=params[base + '-bias'],
                        running_mean=params[base + '-running_mean'],
                        running_var=params[base + '-running_var'],
                        training=mode)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
