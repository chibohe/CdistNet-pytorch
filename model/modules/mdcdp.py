'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 17:36:54
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

from model.modules.transformer import TransformerUnit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MDCDP(nn.Module):
    def __init__(self, n_layer_sae, d_model_sae, d_inner_sae, n_head_sae, d_k_sae, d_v_sae,
                       n_layer, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(MDCDP, self).__init__()
        self.sae = TransformerUnit(n_layer_sae, n_head_sae, d_k_sae, d_v_sae, 
                                    d_model_sae, d_inner_sae, dropout)
        self.cbi_v = TransformerUnit(n_layer, n_head, d_k, d_v, 
                                    d_model, d_inner, dropout)
        self.cbi_s = TransformerUnit(n_layer, n_head, d_k, d_v, 
                                    d_model, d_inner, dropout)
        self.conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1,
                          stride=1, padding=0, bias=False)
        self.active = nn.Sigmoid()
        
    def _generate_pos_mask(self, query, key):
        query_length = query.size(1)
        key_length = key.size(1)
        mask = torch.tril(
            torch.ones((query_length, key_length), dtype=torch.bool)
        ).to(device)
        return mask

    def _generate_tgt_mask(self, query, key):
        pad_mask = (query != self.padding_idx).unsqueeze(1).unsqueeze(3)
        query_length = query.size(1)
        key_length = key.size(1)
        sub_mask = torch.tril(
            torch.ones((query_length, key_length), dtype=torch.bool)
        ).to(device)
        target_mask = pad_mask & sub_mask
        return target_mask

    def forward(self, pos_embedding, vis_feature, sem_embedding):
        # self attention enhancement
        pos_mask = self._generate_pos_mask(pos_embedding, pos_embedding)
        pos_feature = self.sae(pos_embedding, pos_embedding, pos_embedding, pos_mask)
        vis_feature = self.cbi_v(pos_feature, vis_feature, vis_feature)
        sem_mask = self._generate_pos_mask(pos_feature, sem_embedding)
        sem_embedding = self.cbi_s(pos_feature, sem_embedding, sem_embedding, sem_mask)
        # (batch_size, length, channel * 2)
        context = torch.cat([vis_feature, sem_embedding], dim=2)
        context = context.permute(0, 2, 1).unsqueeze(2)
        context = self.conv(context)
        context = context.squeeze(2).permute(0, 2, 1)
        context = self.active(context)
        context_vis_feature = (1 - context) * vis_feature
        context_sem_embedding = context * sem_embedding
        fuse_feature = context_vis_feature + context_sem_embedding

        return fuse_feature
        



        

        
