'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 20:31:36
Contact: yangbitao001@ke.com
'''

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttnLoss(nn.Module):
    def __init__(self, params):
        super(AttnLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, pred, args):
        label, label_length = args['targets'], args['targets_lengths']
        label = label[:, 1:]
        loss = self.loss_func(pred.view(-1, pred.size(-1)), label.contiguous().view(-1))
        return loss
        