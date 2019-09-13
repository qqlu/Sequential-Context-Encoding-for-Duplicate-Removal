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
        # self.encoder_rnn = nn.GRU(3*hidden_size, 3*hidden_size)
        self.encoder_rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        # decoder rnn is list
    
        self.decoder_rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        # encoder shared weights
        self.appear_linear = nn.Linear(1024, hidden_size)
        self.score_linear_1 = nn.Linear(80*32, 512)
        self.score_linear_2 = nn.Linear(512, hidden_size)
        self.box_linear = nn.Linear(80*4, hidden_size)

        self.encoder_feature_linear = nn.Linear(3*hidden_size, hidden_size)
        self.decoder_feature_linear = nn.Linear(hidden_size+64, hidden_size)
        
        # decoder output
        self.out = nn.Linear(2*hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, boxes_feature, boxes_score, boxes_box, all_class_boxes_feature, all_class_boxes_score, all_class_boxes_box, unique_class, unique_class_len):
        hidden_encoder = self.initHidden()
        # hidden_decoder = [0] * 80

        # encoder
        encoder_box_feature = F.relu(self.appear_linear(boxes_feature))
        decoder_box_feature = F.relu(self.appear_linear(all_class_boxes_feature))

        encoder_box_score = F.relu(self.score_linear_2(F.relu(self.score_linear_1(boxes_score))))
        encoder_box_box = F.relu(self.box_linear(boxes_box))

        # encoder_all_feature = encoder_box_score

        encoder_all_feature = torch.cat((encoder_box_feature, encoder_box_score, encoder_box_box), 1)
        decoder_all_feature = torch.cat((decoder_box_feature, all_class_boxes_score, all_class_boxes_box), 1)

        encoder_all_feature_1 = F.relu(self.encoder_feature_linear(encoder_all_feature))
        # print(encoder_all_feature.size())
        # print(decoder_all_feature.size())
        # input()
        decoder_all_feature_1 = F.relu(self.decoder_feature_linear(decoder_all_feature))
        
        encoder_input = torch.unsqueeze(encoder_all_feature_1, 1)
        decoder_input = torch.unsqueeze(decoder_all_feature_1, 1)

        # print(output.size())
        for j in range(self.n_layers):
            # decoder_input = F.relu(encoder_input)
            encoder_output, hidden_encoder = self.encoder_rnn(encoder_input, hidden_encoder)
        
        decoder_output_list = []
        # hidden_decoder = hidden_encoder
        # decoder_rnn = self.decoder_rnn
        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            class_decoder_input = decoder_input[start:end, :, :]
            class_decoder_output, hidden_decoder = self.decoder_rnn(class_decoder_input, hidden_encoder)
        
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