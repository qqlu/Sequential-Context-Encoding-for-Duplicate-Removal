from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from models.GlobalAttention import GlobalAttention
from models.ContextGate import context_gate_factory
from models.rnn_model_full_global_attention_context_gate_output_v3 import Encoder_Decoder
from models.Bridge import Bridge

class pre_post_joint(nn.Module):
    def __init__(self, hidden_size, output_size=1, n_layers=1, use_cuda=True, attn_type='dot', context_type=None, post_score_threshold=0.01):
        super(pre_post_joint, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.output_size = output_size
        self.context_type = context_type
        self.attn_type = attn_type
        self.post_score_threshold = post_score_threshold
        
        self.pre_stage = Encoder_Decoder(self.hidden_size, self.output_size, self.n_layers, self.use_cuda, self.attn_type, self.context_type)
        self.bridge = Bridge(self.post_score_threshold, weight=2)
        self.post_stage = Encoder_Decoder(self.hidden_size, self.output_size, self.n_layers, self.use_cuda, self.attn_type, self.context_type)

    def forward(self, pre_stage_box_feature_variable, pre_stage_box_box_variable, pre_stage_box_score_variable, pre_stage_box_origin_score_variable, pre_stage_box_origin_box_variable, gts_box_tensor, pre_stage_unique_class, pre_stage_unique_class_len):
        # hidden state initialization
        pre_stage_output, pre_stage_last_feature = self.pre_stage(pre_stage_box_feature_variable, pre_stage_box_box_variable, pre_stage_box_score_variable, pre_stage_box_origin_score_variable, pre_stage_unique_class, pre_stage_unique_class_len)
        post_stage_input, post_stage_pre_last_feature = self.bridge(pre_stage_output, pre_stage_last_feature, pre_stage_box_feature_variable, pre_stage_box_box_variable, pre_stage_box_score_variable, pre_stage_box_origin_score_variable, pre_stage_box_origin_box_variable, pre_stage_unique_class, pre_stage_unique_class_len, gts_box_tensor)
        # assign post data
        post_stage_box_feature_variable = post_stage_input[2]
        post_stage_box_box_variable = post_stage_input[3]
        post_stage_box_origin_score_variable = post_stage_input[5]
        post_stage_box_score_variable = post_stage_input[6]
        post_unique_class = post_stage_input[7]
        post_unique_class_len = post_stage_input[8]
        
        post_stage_output, _ = self.post_stage(post_stage_box_feature_variable, post_stage_box_box_variable, post_stage_box_score_variable, post_stage_box_origin_score_variable, post_unique_class, post_unique_class_len, post_stage_pre_last_feature)
        # print(output)
        post_stage_label = post_stage_input[1]
        post_stage_weight = post_stage_input[0]
        post_stage_box_origin_box_tensor = post_stage_input[4]
        return pre_stage_output, post_stage_output, post_stage_label, post_stage_weight, post_stage_box_origin_score_variable, post_stage_box_origin_box_tensor, post_unique_class, post_unique_class_len