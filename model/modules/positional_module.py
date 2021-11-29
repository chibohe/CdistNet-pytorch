'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 15:44:53
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
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class PositionalEmbedding(nn.Module):
    def __init__(self, d_onehot, d_hid, max_seq_len=50, n_position=200):
        super(PositionalEmbedding, self).__init__()
        self.positional_encoding = PositionalEncoding(d_hid, n_position)
        self.embedding = nn.Embedding(max_seq_len + 1, d_onehot)

    def _get_position_index(self, length, batch_size):
        position_index = torch.arange(0, length)
        position_index = position_index.repeat([batch_size, 1])
        position_index = position_index.long()
        return position_index

    def forward(self, input_char):
        batch_size = input_char.size(0)
        length = input_char.size(1)
        # one_hot embedding
        one_hot = self._get_position_index(length, batch_size).to(device)
        one_hot_embedding = self.embedding(one_hot)
        positional_embedding = self.positional_encoding(one_hot_embedding)
        return positional_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()





