'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 15:35:01
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

from model.modules.transformer import ScaledDotProductAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SemanticEmbedding(nn.Module):
    def __init__(self, d_model, num_classes, rnn_layers=2, d_k = 64, max_seq_len=50, 
                rnn_dropout=0, attn_dropout=0.1, padding_idx=1):
        super(SemanticEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, d_model)
        self.sequence_layer = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout)
        self.attention_layer = ScaledDotProductAttention(d_k, attn_dropout)
        self.linear = nn.Linear(d_model, num_classes)
        self.max_seq_len = max_seq_len + 1
        self.padding_idx = padding_idx

    def _generate_mask(self, query, key):
        pad_mask = (query != self.padding_idx).unsqueeze(2)
        query_length = query.size(1)
        key_length = key.size(1)
        sub_mask = torch.tril(
            torch.ones((query_length, key_length), dtype=torch.bool)
        ).to(device)
        target_mask = pad_mask & sub_mask
        return target_mask

    def forward(self, enc_feats, input_char):
        # enc_feats: [n, w, c]
        self.sequence_layer.flatten_parameters()
        input_embedding = self.embedding(input_char)
        semantic_embedding, _ = self.sequence_layer(input_embedding)
        semantic_mask = self._generate_mask(input_char, enc_feats)
        outputs = self.attention_layer(semantic_embedding, enc_feats, enc_feats, semantic_mask)[0]
        return outputs



