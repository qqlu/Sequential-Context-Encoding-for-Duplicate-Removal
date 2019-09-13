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
        self.encoder_rnn = nn.GRU(hidden_size, hidden_size)
        self.decoder_rnn = nn.GRU(hidden_size, hidden_size)

        # shared weights
        self.appear_linear_2 = nn.Linear(1024, hidden_size)
        # self.appear_linear_2 = nn.Linear(512, hidden_size)

        self.axis_score_conv = nn.Conv2d(81, 56, kernel_size=2, stride=1)
        self.axis_score_linear = nn.Linear(56*3*3, hidden_size)

        # decoder output
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, boxes_feature, boxes_box_score, num_proposals):
        hidden_encoder = self.initHidden()
        hidden_decoder = self.initHidden()

        encoder_store = []
        # encoder
        for i in range(num_proposals):
            encoder_box_feature = boxes_feature[i, :]   # 1024
            
            encoder_box_box_score = boxes_box_score[i, :, :, :]  # 81*4*4

            # encoder_box_feature_1 = self.appear_linear_1(encoder_box_feature)
            encoder_box_feature_2 = self.appear_linear_2(encoder_box_feature)

            encoder_axis_score_1 = self.axis_score_conv(encoder_box_box_score.unsqueeze(0))
            # print(axis_score_1.size())
            encoder_axis_score_2 = self.axis_score_linear(encoder_axis_score_1.view(-1))

            encoder_all_feature = torch.add(encoder_box_feature_2, encoder_axis_score_2)
            encoder_output = encoder_all_feature.unsqueeze(0).unsqueeze(0)

            encoder_store.append(encoder_output)
            # print(output.size())
            for j in range(self.n_layers):
                encoder_output, hidden_encoder = self.encoder_rnn(encoder_output, hidden_encoder)
        
        
        # decoder
        output_record = []
        for i in range(num_proposals):
            decoder_output = torch.add(encoder_store[i], encoder_output)

            for j in range(self.n_layers):
                decoder_output = F.relu(decoder_output)
                decoder_output, hidden_decoder = self.decoder_rnn(decoder_output, hidden_decoder)
            output = self.sigmoid(self.out(decoder_output[0]))
            output_record.append(output)
            # print('{}, {}'.format(i, output))
        # print('finished')
        return output_record

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

