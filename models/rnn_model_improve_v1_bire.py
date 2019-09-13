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
        # self.decoder_rnn = nn.GRU(hidden_size, hidden_size)

        # shared weights
        self.appear_linear = nn.Linear(1024, hidden_size)
        self.score_linear_1 = nn.Linear(80*32, 512)
        self.score_linear_2 = nn.Linear(512, hidden_size)
        self.box_linear = nn.Linear(80*4, hidden_size)

        self.all_feature_linear = nn.Linear(3*hidden_size, hidden_size)

        # decoder output
        self.out = nn.Linear(2*hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, boxes_feature, boxes_box_score, boxes_box, num_proposals):
        hidden_encoder = self.initHidden()
        # hidden_decoder = self.initHidden()

        # encoder
        encoder_box_feature = F.relu(self.appear_linear(boxes_feature))

        encoder_box_score = F.relu(self.score_linear_2(F.relu(self.score_linear_1(boxes_box_score))))
        encoder_box_box = F.relu(self.box_linear(boxes_box))

        # encoder_all_feature = encoder_box_score

        encoder_all_feature = torch.cat((encoder_box_feature, encoder_box_score, encoder_box_box), 1)
        encoder_all_feature_1 = F.relu(self.all_feature_linear(encoder_all_feature))
        encoder_input = torch.unsqueeze(encoder_all_feature_1, 1)
        # print(output.size())
        for j in range(self.n_layers):
            # decoder_input = F.relu(encoder_input)
            encoder_output, hidden_encoder = self.encoder_rnn(encoder_input, hidden_encoder)
            hidden_decoder = hidden_encoder
            decoder_output, hidden_decoder = self.encoder_rnn(encoder_input, hidden_decoder)
            
        output = self.sigmoid(self.out(decoder_output))
        # print(output)
        return output

    def initHidden(self):
        result = Variable(torch.zeros(2, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result