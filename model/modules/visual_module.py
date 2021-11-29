'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 15:24:32
Contact: yangbitao001@ke.com
'''


# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.transformer import TransformerUnit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class VisualModule(nn.Module):
    def __init__(self, d_input, layers, n_layer, d_model, d_inner, 
                    n_head, d_k, d_v, dropout=0.1):
        super(VisualModule, self).__init__()
        self.block = BasicBlock
        self.resnet = ResNet(d_input, d_model, self.block, layers)
        self.transformer = TransformerUnit(n_layer, n_head, d_k, d_v, 
                                    d_model, d_inner, dropout)

    def forward(self, x):
        x = self.resnet(x)
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.transformer(x, x, x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def _conv3x3(self, inplanes, planes):
        return nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                         padding=1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()
        self.output_channel_blocks = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]
        self.inplanes = int(output_channel / 8)

        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16), 
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.relu = nn.ReLU(inplace=True)
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), int(output_channel / 8),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(int(output_channel / 8))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_blocks[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_blocks[0], self.output_channel_blocks[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_blocks[0])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_blocks[1], layers[1])
        self.conv2 = nn.Conv2d(self.output_channel_blocks[1], self.output_channel_blocks[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_blocks[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_blocks[2], layers[2])
        self.conv3 = nn.Conv2d(self.output_channel_blocks[2], self.output_channel_blocks[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_blocks[2])
        self.layer4 = self._make_layer(block, self.output_channel_blocks[3], layers[3])
        self.conv4 = nn.Conv2d(self.output_channel_blocks[3], self.output_channel_blocks[3],
                               kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(self.output_channel_blocks[3])
        self.conv5 = nn.Conv2d(self.output_channel_blocks[3], self.output_channel_blocks[3],
                               kernel_size=(2, 2), stride=(1, 1), padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(self.output_channel_blocks[3])

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes
        for i in range(blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x



