import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class conv_qkv(nn.Conv2d):
    """Conv1d for Q, K, V"""


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(SelfAttention2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.group_channels = out_channels // groups

        self.conv_qkv = conv_qkv(in_channels, out_channels * 3, kernel_size=1, bias=False)
        self.unfold = nn.Unfold(kernel_size, 1, padding, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_h = nn.Parameter(torch.randn(self.group_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.relative_w = nn.Parameter(torch.randn(self.group_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        N, C, H, W = x.shape

        qkv = self.conv_qkv(x)
        q, kv = qkv.split([self.out_channels, self.out_channels * 2], dim=1)
        kv = self.unfold(kv).reshape(N, self.out_channels * 2, self.kernel_size**2, H * W).transpose(2, 3).reshape(N, self.out_channels * 2, H, W, self.kernel_size, self.kernel_size)
        k, v = kv.split(self.out_channels, dim=1)
        k_h, k_w = k.reshape(N, self.groups, self.group_channels, H, W, self.kernel_size, self.kernel_size).split(self.group_channels // 2, dim=2)
        kr = torch.cat((k_h + self.relative_h, k_w + self.relative_w), dim=2)

        q = q.view(N, self.groups, self.group_channels, H, W, 1)
        kr = kr.view(N, self.groups, self.group_channels, H, W, -1)
        v = v.view(N, self.groups, self.group_channels, H, W, -1) # N, g, C//g, H, W, kxk

        qkr = (q * kr).sum(dim=2)
        qkr = self.softmax(qkr).unsqueeze(2) # N, g, 1, H, W, kxk
        out = (qkr * v).sum(dim=-1).reshape(N, self.out_channels, H, W)

        if self.stride > 1:
            out = self.pooling(out)

        return out

    def reset_parameters(self):
        n = self.in_channels
        self.conv_qkv.weight.data.normal_(0, math.sqrt(1. / n))
        n = self.group_channels
        nn.init.normal_(self.relative_h, 0, math.sqrt(1. / n))
        nn.init.normal_(self.relative_w, 0, math.sqrt(1. / n))


class SelfAttentionBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SelfAttentionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = SelfAttention2d(width, width, kernel_size=7, stride=stride, padding=3, groups=8)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SelfAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(SelfAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, conv_qkv):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SelfAttentionBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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


def sa_resnet50(**kwargs):
    model = SelfAttentionNet(SelfAttentionBlock, [3, 4, 6, 3], **kwargs)
    return model

