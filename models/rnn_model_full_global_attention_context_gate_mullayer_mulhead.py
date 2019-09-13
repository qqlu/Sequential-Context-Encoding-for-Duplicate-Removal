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

class Encoder_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size=1, n_layers=1, use_cuda=True, attn_type='dot', multi_head=1, context_type=None):
        super(Encoder_Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.output_size = output_size
        self.bidirectional = True
        self.multi_head = multi_head

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # encoder
        self.appear_linear = nn.Linear(1024, hidden_size)
        self.feature_linear = nn.Linear(hidden_size+96, hidden_size)
        self.encoder_rnn = nn.ModuleList([nn.GRU(hidden_size, hidden_size, self.n_layers, bidirectional=self.bidirectional) for i in range(80)])
        
        # decoder_rnn
        self.decoder_rnn = nn.ModuleList([nn.GRU(hidden_size, hidden_size, self.n_layers, bidirectional=self.bidirectional) for i in range(80)])
        self.out = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # attention
        self.atten = nn.ModuleList([GlobalAttention(hidden_size*2, attn_type=attn_type) for i in range(self.multi_head)])
        self.reduction = nn.Linear(hidden_size*2*self.multi_head, hidden_size*2)
        
        # self.atten = GlobalAttention(hidden_size*2, attn_type=attn_type)
        # context
        self.context_type = context_type
        if self.context_type is not None:
            self.context_gate = context_gate_factory(self.context_type, hidden_size, hidden_size*2, hidden_size*2, hidden_size*2)

    def forward(self, all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, unique_class, unique_class_len):
        # hidden state initialization
        hidden_encoder = self.initHidden()
        # hidden_decoder = self.initHidden(self.hidden_size)
        
        # encoder
        encoder_box_feature = F.relu(self.appear_linear(all_class_box_feature_variable))
        encoder_all_feature = torch.cat((encoder_box_feature, all_class_box_score_variable, all_class_box_box_variable, all_class_box_origin_score_variable), 1)
        encoder_all_feature_1 = F.relu(self.feature_linear(encoder_all_feature))
        encoder_input = torch.unsqueeze(encoder_all_feature_1, 1)

        #encoder result
        final_output_list = []
        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            class_encoder_input = encoder_input[start:end, :, :]
            class_encoder_output, class_encoder_hidden = self.encoder_rnn[j](class_encoder_input, hidden_encoder) 
            class_decoder_output, class_decoder_hidden = self.encoder_rnn[j](class_encoder_input, class_encoder_hidden) 
            class_final_mulhead_output = []
            # print(class_decoder_output.size())
            for qq in range(self.multi_head):
                class_final_mulhead_output_temp, _ = self.atten[qq](class_decoder_output.permute(1, 0, 2), class_encoder_output.permute(1, 0, 2)) 
                # print(class_final_mulhead_output_temp.size())
                class_final_mulhead_output.append(class_final_mulhead_output_temp)
            # for qq in range(len(class_final_mulhead_output)):
            #     print(class_final_mulhead_output[qq].size())
            class_final_output = torch.cat(class_final_mulhead_output, 2)
            class_final_output = self.reduction(class_final_output)
            # print(class_final_output.size())
            # input()
            if self.context_type is not None:
                class_final_output = self.context_gate(class_encoder_input.view(-1, class_encoder_input.size(2)), class_decoder_output.view(-1, class_decoder_output.size(2)), class_final_output.view(-1, class_final_output.size(2)))
                class_final_output = class_final_output.view(-1, 1, self.hidden_size*2)
            
            final_output_list.append(class_final_output)

        # decoder
        final_output = torch.cat(final_output_list)
        # print(final_output.size())
        # print(encoder_input.size())
        # input()
        output = self.sigmoid(self.out(final_output))
        # print(output)
        return output

    def initHidden(self):
        if self.bidirectional:
            result = Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size))
        else:
            result = Variable(torch.zeros(self.n_layers * 1, 1, self.hidden_size))

        if self.use_cuda:
            return result.cuda()
        else:
            return result