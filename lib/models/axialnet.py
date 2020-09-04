import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


__all__ = ['axial50s', 'axial50m', 'axial50l']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.q_transform = q_transform(in_planes, out_planes // 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.k_transform = k_transform(in_planes, out_planes // 2, kernel_size=1, stride=1,
                                       padding=0, bias=False)
        self.v_transform = v_transform(in_planes, out_planes, kernel_size=1, stride=1,
                                       padding=0, bias=False)

        self.bn_q = nn.BatchNorm2d(out_planes // 2)
        self.bn_k = nn.BatchNorm2d(out_planes // 2)
        self.bn_v = nn.BatchNorm2d(out_planes)

        self.bn_qk = nn.BatchNorm2d(groups)
        self.bn_qr = nn.BatchNorm2d(groups)
        self.bn_kr = nn.BatchNorm2d(groups)

        self.bn_sv = nn.BatchNorm2d(out_planes)
        self.bn_sve = nn.BatchNorm2d(out_planes)

        # Position embedding
        # Note: Instead of creating an absolute spatial grid and slicing into it
        # in every forward call (readable code but slow execution) create a compact
        # positional embedding parameter that assumes axial translation on axis 1
        self.q_relative = nn.Parameter(torch.randn(self.group_planes // 2, kernel_size), requires_grad=True)
        self.k_relative = nn.Parameter(torch.randn(self.group_planes // 2, kernel_size), requires_grad=True)
        self.v_relative = nn.Parameter(torch.randn(self.group_planes, kernel_size), requires_grad=True)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.transpose(2, 3)
        N, C, H, W = x.shape

        # Transformations
        q = self.q_transform(x)
        q = self.bn_q(q)
        k = self.k_transform(x)
        k = self.bn_k(k)
        v = self.v_transform(x)
        v = self.bn_v(v)

        # Calculate position embedding
        q_embedding = self.q_relative.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        k_embedding = self.k_relative.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        v_embedding = self.v_relative.unsqueeze(-1).expand(-1, -1, self.kernel_size)


        qr = torch.einsum('bgciw, cij->bgijw', q.reshape(N, self.groups, self.group_planes // 2, H, W), q_embedding)
        qr = self.bn_qr(qr.reshape(N, self.groups, -1, W)).reshape(N, self.groups, H, H, W)

        kr = torch.einsum('bgciw, cij->bgijw', k.reshape(N, self.groups, self.group_planes // 2, H, W), k_embedding)
        kr = self.bn_kr(kr.reshape(N, self.groups, -1, W)).reshape(N, self.groups, H, H, W)
        kr = kr.transpose(2, 3)

        # Blocks of axial attention
        q = q.reshape(N, self.groups, self.group_planes // 2, H, W)
        k = k.reshape(N, self.groups, self.group_planes // 2, H, W)

        # (q, k)
        qk = torch.einsum('bgciw, bgcjw->bgijw', q, k)
        qk = self.bn_qk(qk.reshape(N, self.groups, -1, W)).reshape(N, self.groups, H, H, W)

        # (N, groups, H, H, W)
        similarity = F.softmax(qk + qr + kr, dim=3)
        sv = torch.einsum('bgijw, bgcjw->bgciw', similarity, v.reshape(N, self.groups, self.group_planes, H, W))
        sve = torch.einsum('bgijw, cji->bgciw', similarity, v_embedding)
        output = self.bn_sv(sv.reshape(N, -1, H, W)) + self.bn_sve(sve.reshape(N, -1, H, W))

        if self.width:
            output = output.transpose(2, 3)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        n = self.in_planes * self.group_planes
        self.q_transform.weight.data.normal_(0, math.sqrt(1. / n))
        n = self.in_planes
        self.k_transform.weight.data.normal_(0, math.sqrt(1. / n))
        self.v_transform.weight.data.normal_(0, math.sqrt(1. / n))
        n = self.out_planes // 2
        nn.init.normal_(self.q_relative, 0, math.sqrt(1. / n))
        nn.init.normal_(self.k_relative, 0, math.sqrt(1. / n))
        n = self.out_planes
        nn.init.normal_(self.v_relative, 0, math.sqrt(1. / n))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=56)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=56,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=28,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=14,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * block.expansion * s), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, q_transform) or isinstance(m, k_transform) or isinstance(m, v_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def axial50s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def axial50m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.75, **kwargs)
    return model


def axial50l(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model
