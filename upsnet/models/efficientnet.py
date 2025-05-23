
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import warnings
from upsnet.operators.modules.deform_conv import DeformConv
from upsnet.config.config import config
import torch.utils.checkpoint

if not config.network.backbone_fix_bn and config.network.use_syncbn:
    from upsnet.operators.modules.distbatchnorm import BatchNorm2d
    nn.BatchNorm2d = BatchNorm2d

def get_params(model, prefixs, suffixes, exclude=None):
    for name, module in model.named_modules():
        for prefix in prefixs:
            if name == prefix:
                for n, p in module.named_parameters():
                    n = '.'.join([name, n])
                    if type(exclude) == list and n in exclude:
                        continue
                    if type(exclude) == str and exclude in n:
                        continue
                    for suffix in suffixes:
                        if (n.split('.')[-1].startswith(suffix) or n.endswith(suffix)) and p.requires_grad:
                            yield p
                break

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fix_bn=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if fix_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            for i in self.bn1.parameters():
                i.requires_grad = False
            for i in self.bn2.parameters():
                i.requires_grad = False
            for i in self.bn3.parameters():
                i.requires_grad = False

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class DCNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fix_bn=True, deformable_group=1):
        super(DCNBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_offset = nn.Conv2d(planes, 18 * deformable_group, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_offset.weight.data.zero_()
        self.conv2_offset.bias.data.zero_()
        self.conv2 = DeformConv(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if fix_bn:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()
            for i in self.bn1.parameters():
                i.requires_grad = False
            for i in self.bn2.parameters():
                i.requires_grad = False
            for i in self.bn3.parameters():
                i.requires_grad = False

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        offset = self.conv2_offset(out)
        out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class EfficientStem(nn.Module):
    def __init__(self):
        super(EfficientStem, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion=6, dilation=1, use_deform=False, fix_bn=True):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.use_expand = expansion != 1
        mid_channels = in_channels * expansion if self.use_expand else in_channels
        self.relu = nn.ReLU(inplace=True)

        if self.use_expand:
            self.expand_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(mid_channels)
            if fix_bn:
                self.bn0.eval()
                for p in self.bn0.parameters():
                    p.requires_grad = False

        if use_deform:
            self.offset_conv = nn.Conv2d(mid_channels, 3 * 3 * 2, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
            nn.init.constant_(self.offset_conv.weight, 0)
            nn.init.constant_(self.offset_conv.bias, 0)
            self.dw_conv = DeformConv(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        else:
            self.dw_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        if fix_bn:
            self.bn1.eval()
            for p in self.bn1.parameters():
                p.requires_grad = False

        self.project_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if fix_bn:
            self.bn2.eval()
            for p in self.bn2.parameters():
                p.requires_grad = False

    def forward(self, x):
        identity = x
        out = x
        if self.use_expand:
            out = self.expand_conv(out)
            out = self.bn0(out)
            out = self.relu(out)
        if hasattr(self, 'offset_conv'):
            offset = self.offset_conv(out)
            out = self.dw_conv(out, offset)
        else:
            out = self.dw_conv(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.project_conv(out)
        out = self.bn2(out)
        if self.use_res_connect:
            out = out + identity
        return self.relu(out)

class EfficientNetBackbone(nn.Module):
    def __init__(self, blocks):
        super(EfficientNetBackbone, self).__init__()
        self.fix_bn = config.network.backbone_fix_bn
        self.with_dilation = config.network.backbone_with_dilation
        self.with_dpyramid = config.network.backbone_with_dpyramid
        self.with_dconv = config.network.backbone_with_dconv
        self.freeze_at = config.network.backbone_freeze_at

        self.conv1 = EfficientStem()
        self.res2 = self._make_stage(MBConvBlock, in_channels=64, out_channels=256, blocks=blocks[0], stride=1, stage=2)
        self.res3 = self._make_stage(MBConvBlock, in_channels=256, out_channels=512, blocks=blocks[1], stride=2, stage=3)
        self.res4 = self._make_stage(MBConvBlock, in_channels=512, out_channels=1024, blocks=blocks[2], stride=2, stage=4)
        if self.with_dilation:
            res5_stride, res5_dilation = 1, 2
        else:
            res5_stride, res5_dilation = 2, 1
        self.res5 = self._make_stage(MBConvBlock, in_channels=1024, out_channels=2048, blocks=blocks[3], stride=res5_stride, stage=5, dilation=res5_dilation)

        if self.freeze_at > 0:
            for p in self.conv1.parameters():
                p.requires_grad = False
            self.conv1.eval()
            for i in range(2, self.freeze_at + 1):
                getattr(self, 'res{}'.format(i)).eval()
                for p in getattr(self, 'res{}'.format(i)).parameters():
                    p.requires_grad = False

    def _make_stage(self, block_fn, in_channels, out_channels, blocks, stride, stage, dilation=1):
        layers = []
        use_deform = (self.with_dconv <= stage)
        layers.append(block_fn(in_channels, out_channels, stride=stride, expansion=6, dilation=dilation, use_deform=use_deform, fix_bn=self.fix_bn))
        for i in range(1, blocks):
            use_def = use_deform or (self.with_dpyramid and i == blocks - 1)
            layers.append(block_fn(out_channels, out_channels, stride=1, expansion=6, dilation=dilation, use_deform=use_def, fix_bn=self.fix_bn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        res2 = self.res2(x)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        if self.freeze_at == 1:
            x = x.detach()
        if self.freeze_at == 2:
            res2 = res2.detach()
        if self.freeze_at == 3:
            res3 = res3.detach()
        if self.freeze_at == 4:
            res4 = res4.detach()
        if self.freeze_at == 5:
            res5 = res5.detach()
        return res2, res3, res4, res5