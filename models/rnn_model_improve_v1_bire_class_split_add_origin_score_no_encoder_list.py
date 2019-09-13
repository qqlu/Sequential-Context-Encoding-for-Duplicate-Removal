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

class Encoder_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size=1, n_layers=1, use_cuda=True):
        super(Encoder_Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.output_size = output_size

        # rnn model
        # self.encoder_rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn = nn.ModuleList([nn.GRU(hidden_size, hidden_size, bidirectional=True) for i in range(80)])
        
        # encoder shared weights
        self.appear_linear = nn.Linear(1024, hidden_size)
        self.score_linear_1 = nn.Linear(80*32, 512)
        self.score_linear_2 = nn.Linear(512, hidden_size)
        self.box_linear = nn.Linear(80*4, hidden_size)

        # self.encoder_feature_linear = nn.Linear(3*hidden_size, hidden_size)
        self.decoder_feature_linear = nn.Linear(hidden_size+96, hidden_size)
        
        # decoder output
        self.out = nn.Linear(2*hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, boxes_feature, boxes_score, boxes_box, all_class_boxes_feature, all_class_boxes_score, all_class_boxes_box, all_class_boxes_origin_score, unique_class, unique_class_len):
        hidden_encoder = self.initHidden()

        # encoder
        decoder_box_feature = F.relu(self.appear_linear(all_class_boxes_feature))

        decoder_all_feature = torch.cat((decoder_box_feature, all_class_boxes_score, all_class_boxes_box, all_class_boxes_origin_score), 1)

        decoder_all_feature_1 = F.relu(self.decoder_feature_linear(decoder_all_feature))
        
        decoder_input = torch.unsqueeze(decoder_all_feature_1, 1)
        
        decoder_output_list = []
        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            class_decoder_input = decoder_input[start:end, :, :]     
            class_decoder_output, hidden_decoder = self.decoder_rnn[j](class_decoder_input, hidden_encoder)
            decoder_output_list.append(class_decoder_output)
        decoder_output = torch.cat(decoder_output_list)

        output = self.sigmoid(self.out(decoder_output))
        # print(output)
        return output

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result