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
        self.rnn_stage1 = nn.GRU(hidden_size, hidden_size)
        # self.decoder_rnn_class = nn.GRU(hidden_size, hidden_size)
        # self.encoder_rnn_stage2 = nn.GRU(2*hidden_size, hidden_size)
        self.rnn_stage2 = nn.GRU(2*hidden_size, hidden_size)
        # shared weights
        self.appear_linear = nn.Linear(1024, hidden_size)
        self.score_linear_1 = nn.Linear(80*32, 512)
        self.score_linear_2 = nn.Linear(512, hidden_size)
        self.box_linear = nn.Linear(80*4, hidden_size)

        self.all_feature_linear = nn.Linear(3*hidden_size, hidden_size)

        # decoder output
        self.out_class = nn.Linear(hidden_size, output_size)
        self.out_final = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, boxes_feature, boxes_box_score, boxes_box, num_proposals):
        hidden_stage1, hidden_stage2 = self.initHidden()
        # hidden_decoder = self.initHidden()

        # encoder
        encoder_box_feature = F.relu(self.appear_linear(boxes_feature))

        encoder_box_score = F.relu(self.score_linear_2(self.score_linear_1(boxes_box_score)))
        encoder_box_box = F.relu(self.box_linear(boxes_box))

        # encoder_all_feature = encoder_box_score

        encoder_all_feature = torch.cat((encoder_box_feature, encoder_box_score, encoder_box_box), 1)
        encoder_all_feature_1 = F.relu(self.all_feature_linear(encoder_all_feature))
        encoder_input = torch.unsqueeze(encoder_all_feature_1, 1)

        encoder_output, hidden_encoder = self.rnn_stage1(encoder_input, hidden_stage1)
        hidden_decoder_stage1 = hidden_encoder
        decoder_output_class, hidden_decoder_stage1 = self.rnn_stage1(encoder_input, hidden_decoder_stage1)
        
        # hidden_decoder_final = hidden_encoder
        input_stage2 = torch.cat((encoder_input, decoder_output_class), 2)
        decoder_output_final, hidden_decoder_final = self.rnn_stage2(input_stage2, hidden_stage2)
        hidden_decoder_stage2 = hidden_decoder_final
        decoder_output_final, hidden_decoder_stage2 = self.rnn_stage2(input_stage2, hidden_decoder_stage2)

        output_class = self.sigmoid(self.out_class(decoder_output_class))
        output_final = self.sigmoid(self.out_final(decoder_output_final))
        # print(output)
        return [output_class, output_final]

    def initHidden(self):
        result_stage1 = Variable(torch.zeros(1, 1, self.hidden_size))
        result_stage2 = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result_stage1.cuda(), result_stage2.cuda()
        else:
            return result_stage1, result_stage2

