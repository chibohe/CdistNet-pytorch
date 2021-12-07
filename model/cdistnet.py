'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 17:33:42
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
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.visual_module import VisualModule
from model.modules.positional_module import PositionalEmbedding
from model.modules.semantic_module import SemanticEmbedding
from model.modules.mdcdp import MDCDP

from character import LabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CDistNet(nn.Module):
    def __init__(self, flags):
        super(CDistNet, self).__init__()
        global_flags = flags.Global
        d_input = global_flags.image_shape[0]
        vis_flags = flags.VisualModule
        self.vis_module = VisualModule(d_input, vis_flags.layers, vis_flags.n_layer, 
                                        vis_flags.d_model, vis_flags.d_inner, 
                                        vis_flags.n_head, vis_flags.d_k, vis_flags.d_v, 
                                        vis_flags.dropout)
        pos_flags = flags.PositionalEmbedding
        self.pos_module = PositionalEmbedding(pos_flags.d_onehot, pos_flags.d_hid, 
                                        global_flags.batch_max_length, pos_flags.n_position)
        sem_flags = flags.SemanticEmbedding
        self.converter = LabelConverter(flags)
        self.sem_module = SemanticEmbedding(sem_flags.d_model, self.converter.char_num,
                                        sem_flags.rnn_layers, sem_flags.d_k, global_flags.batch_max_length, 
                                        sem_flags.rnn_dropout, sem_flags.attn_dropout, global_flags.padding_idx)
        mdcdp_flags = flags.MDCDP
        self.mdcdp = MDCDP(mdcdp_flags.n_layer_sae, mdcdp_flags.d_model_sae, mdcdp_flags.d_inner_sae, 
                            mdcdp_flags.n_head_sae, mdcdp_flags.d_k_sae, mdcdp_flags.d_v_sae, 
                            mdcdp_flags.n_layer, mdcdp_flags.d_model, mdcdp_flags.d_inner, 
                            mdcdp_flags.n_head, mdcdp_flags.d_k, mdcdp_flags.d_v,mdcdp_flags.dropout)
        self.linear = nn.Linear(mdcdp_flags.d_model, self.converter.char_num)
        self.is_train = flags.Global.is_train
        self.max_seq_len = global_flags.batch_max_length + 1


    def forward(self, input, input_char):
        vis_feature = self.vis_module(input)
        pos_embedding = self.pos_module(input_char)
        if self.is_train:
            sem_embedding = self.sem_module(vis_feature, input_char)
            outputs = self.mdcdp(pos_embedding, vis_feature, sem_embedding)
            outputs = self.linear(outputs)
        else:
            outputs = []
            for i in range(self.max_seq_len):
                sem_embedding = self.sem_module(vis_feature, input_char)
                fuse_feature = self.mdcdp(pos_embedding, vis_feature, sem_embedding)
                fuse_feature_step = fuse_feature[:, i, :]
                fuse_feature_step = self.linear(fuse_feature_step)
                outputs.append(fuse_feature_step)
                fuse_feature_step = F.softmax(fuse_feature_step, dim=-1)
                _, max_idx = torch.max(fuse_feature_step, dim=1, keepdim=False)
                if i < self.max_seq_len - 1:
                    input_char[:, i+1] = max_idx
            outputs = torch.stack(outputs, dim=1)

        return outputs




if __name__ == "__main__":
    input = torch.randn(2, 3, 128, 32).to(device)
    input_char = torch.randint(100, size=(2, 20)).to(device)
    cdistnet = CDistNet().to(device)
    output = cdistnet(input, input_char)
    print(output.shape)








