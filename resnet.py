import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import random
import numpy as np 

import copy

__all__ = ['ResNet18', 'ResNet50']

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import random
import numpy as np 

import copy


C_DIM=10
C_Start_Layer=4000
feature_store=[]
out_store=[]
feature_store1=[]
out_store1=[]
layer_number=0
out_store_tmp=[]
is_first_input=1
noise_threshold1=0.51
noise_threshold2=0.0001
noise_threshold3=0.0001
noise_ratio = 1.0
noise_increment=0.5



def perturb_inp(x):
    return x

def perturb(x):
    return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )


    def forward(self, x):
        global feature_store
        global layer_number
        global out_store
        global out_store1
        global is_first_input
        layer_number=layer_number+1

        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        if is_first_input==0:
            out = perturb(out, layer_number)
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        global feature_store
        global layer_number
        global out_store
        global out_store1
        global is_first_input
        layer_number=layer_number+1


        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))


        if is_first_input==0:
            out = perturb(out, layer_number)
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, n_threshold1, n_threshold2, n_threshold3, n_increment, n_ratio):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.n_threshold1 = n_threshold1
        self.n_threshold2 = n_threshold2
        self.n_threshold3 = n_threshold3
        self.n_increment = n_increment
        self.n_ratio = n_ratio

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        global feature_store
        global layer_number
        global out_store
        global feature_store1
        global out_store1
        global is_first_input
        global noise_threshold1
        global noise_threshold2
        global noise_threshold3
        global noise_ratio 
        global noise_increment
        noise_threshold1 = self.n_threshold1
        noise_threshold2 = self.n_threshold2
        noise_threshold2 = self.n_threshold3
        noise_increment = self.n_increment
        noise_ratio = self.n_ratio

        is_first_input=0
        feature_store = []
        out_store = []
        feature_store1 = []
        out_store1 = []
        layer_number = 0

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        outf = self.linear(out)
        feature = outf


        return out_store, feature_store, feature, outf


def ResNet18(classes=10, noise_threshold1=0, noise_threshold2=0, noise_threshold3=0, noise_increment=0, noise_ratio=0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=classes, n_threshold1=noise_threshold1, n_threshold2=noise_threshold2, n_threshold3=noise_threshold3, n_increment=noise_increment, n_ratio=noise_ratio)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(classes=10, noise_threshold1=0, noise_threshold2=0, noise_threshold3=0, noise_increment=0, noise_ratio=0):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=classes, n_threshold1=noise_threshold1, n_threshold2=noise_threshold2, n_threshold3=noise_threshold3, n_increment=noise_increment, n_ratio=noise_ratio)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152(): 
    return ResNet(Bottleneck, [3, 8, 36, 3])


def ResNet1202():
    return ResNet(BasicBlock, [50, 50, 50, 3])



