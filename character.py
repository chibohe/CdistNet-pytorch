'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 20:29:46
Contact: yangbitao001@ke.com
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LabelConverter(object):
    def __init__(self, flags):
        # character (str): set of the possible characters.
        flags = flags.Global
        self.character_type = flags.character_type
        if self.character_type == 'en':
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif self.character_type == 'ch':
            character_dict_path = flags.character_dict_path
            add_space = False
            if hasattr(flags, 'use_space_char'):
                add_space = flags.use_space_char
            self.character_str = ""
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if add_space:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif self.character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            self.character_str = None
        assert self.character_str is not None, \
            "Nonsupport type of the character: {}".format(self.character_str)

        self.character = ['[GO]', '[END]'] + dict_character  #  '[Go]' for the start token, '[s]' for the end token
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i 
        self.char_num = len(self.character)

    def encode(self, text):
        length = [len(s) for s in text]
        batch_max_length = max(length) + 2
        batch_size = len(length)
        outputs = torch.LongTensor(batch_size, batch_max_length).fill_(1)
        for i in range(batch_size):
            curr_text = ['[GO]']
            curr_text.extend(list(text[i]))
            curr_text.append('[END]')
            curr_text = [self.dict[char] for char in curr_text]
            outputs[i, :len(curr_text)] = torch.LongTensor(curr_text)
        return (outputs, torch.IntTensor(length))

    def decode(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text_conf = []
        for idx, prob in zip(preds_idx, preds_prob):
            curr_text = [self.character[index] for index in idx]
            text_conf.append((curr_text, prob))
        result_list = []
        for text, prob in text_conf:
            end_index = ''.join(text).find('[END]')
            text = text[: end_index]
            prob = prob[: end_index]
            result_list.append((''.join(text), prob))
        return result_list