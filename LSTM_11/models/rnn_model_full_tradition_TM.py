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

class Encoder_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size=1, n_layers=1, use_cuda=True):
        super(Encoder_Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.output_size = output_size
        self.teacher_forcing_ratio  = 0.5
        self.decoder_input_dim = 32
        # encoder
        self.encoder_appear_linear = nn.Linear(1024, hidden_size)
        self.encoder_feature_linear = nn.Linear(hidden_size+96, hidden_size)
        self.encoder_rnn = nn.ModuleList([nn.GRU(hidden_size, hidden_size, bidirectional=True) for i in range(80)])

        # decoder_rnn
        self.decoder_rnn = nn.ModuleList([nn.GRU(self.decoder_input_dim, 4) for i in range(80)])
        self.out = nn.Linear(2*hidden_size, 4)
        self.sigmoid = nn.Sigmoid()
        self.encoder_EOS = torch.unsqueeze(torch.ones(1, hidden_size), 1)
        
        # decoder embedding, 0--0, 1--1, sos--2, eos--3
        self.decoder_embedding = nn.Embedding(4, self.decoder_input_dim)
        self.decoder_EOS = self.decoder_embedding(Variable(torch.LongTensor([[3]]))).cuda()
        self.decoder_SOS = self.decoder_embedding(Variable(torch.LongTensor([[2]]))).cuda()

    def forward(self, all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, all_class_box_label_variable, all_class_box_weight_variable, unique_class, unique_class_len):
        # hidden state initialization
        hidden_encoder = self.initHidden(self.hidden_size)
        # hidden_decoder = self.initHidden(self.hidden_size)
        
        # encoder
        encoder_box_feature = F.relu(self.appear_linear(all_class_box_feature_variable))
        encoder_all_feature = torch.cat((encoder_box_feature, all_class_box_score_variable, all_class_box_box_variable, all_class_box_origin_score_variable), 1)
        encoder_all_feature_1 = F.relu(self.encoder_feature_linear(encoder_all_feature))
        encoder_input = torch.unsqueeze(encoder_all_feature_1, 1)

        #encoder result
        encoder_output_list = []
        encoder_hidden_list = []
        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            class_encoder_input = torch.cat((decoder_input[start:end, :, :], encoder_EOS), 0)
            class_encoder_output, class_encoder_hidden = self.encoder_rnn[j](class_encoder_input, hidden_encoder) 
            encoder_output_list.append(class_encoder_output)
            encoder_hidden_list.append(class_encoder_hidden)

        # decoder
        all_class_decoder_input = self.decoder_embedding(all_class_box_label_variable)
        decoder_output_list = []
        decoder_label_list = []
        decoder_weight_list = []

        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            class_decoder_output_list = []
            if use_teacher_forcing:
                class_decoder_output_ini, class_decoder_hidden = self.decoder_rnn[j](self.decoder_SOS, encoder_hidden_list[j])
                class_decoder_output, class_decoder_hidden = self.decoder_rnn[j](all_class_decoder_input[start:end, :, :], class_decoder_hidden)
                class_decoder_output_list.append(torch.cat((class_decoder_output_ini, class_decoder_output), 0))
                class_decoder_label_list.append(torch.cat((all_class_box_)))        
            else:
                class_decoder_output, class_decoder_hidden = self.decoder_rnn[j](self.decoder_SOS, encoder_hidden_list[j])
                topv, topi= class_decoder_output.data.topk(1)
                ni = topi[0][0]
                class_decoder_input = self.decoder_embedding(Variable(torch.LongTensor([[ni]])).cuda())
                class_decoder_output_list.append(class_decoder_output)
                if ni==3:
                    class_decoder_label_list = all_class_box_label_variable[0, start:start+1]
                    class_decoder_weight_list = all_class_box_weight_variable[0, start:start+1]
                    break
                for jj in range(start, end):
                    class_decoder_output, class_decoder_hidden = self.decoder_rnn[j](class_decoder_input, class_decoder_hidden)
                    topv, topi= class_decoder_output.data.topk(1)
                    ni = topi[0][0]
                    class_decoder_input = self.decoder_embedding(Variable(torch.LongTensor([[ni]])).cuda())
                    class_decoder_output_list.append(class_decoder_output)
                    if ni==3:
                        class_decoder_label_list = all_class_box_label_variable[0, start:jj]
                        class_decoder_weight_list = all_class_box_weight_variable[0, start:jj]
                        break
            decoder_output_list.append(torch.cat(class_decoder_output_list))
            decoder_label_list.append(class_decoder_label_list)
            decoder_weight_list.append(class_decoder_weight_list)
        decoder_output = torch.cat(decoder_output_list)

        output = F.log_softmax(self.out(decoder_output))
        label = torch.cat(decoder_label_list)
        weight = torch.cat(decoder_weight_list)
        # print(output)
        return output, label, weight 

    def initHidden(hidden_size):
        result = Variable(torch.zeros(2, 1, hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result