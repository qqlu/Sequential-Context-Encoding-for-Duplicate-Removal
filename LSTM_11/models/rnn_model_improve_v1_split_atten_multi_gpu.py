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
        self.encoder_rnn = nn.GRU(hidden_size, hidden_size)
        self.decoder_rnn = nn.GRU(hidden_size, hidden_size)

        # attention
        self.attn = nn.Linear(int(hidden_size*2), 1)
        self.attn_combine = nn.Linear(int(hidden_size*2), int(hidden_size))

        # shared weights
        self.appear_linear = nn.Linear(1024, hidden_size)
        self.score_linear_1 = nn.Linear(80*32, 512)
        self.score_linear_2 = nn.Linear(512, hidden_size)
        self.box_linear = nn.Linear(80*4, hidden_size)

        self.all_feature_linear = nn.Linear(3*hidden_size, hidden_size)

        # decoder output
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, boxes_feature, boxes_box_score, boxes_box, boxes_label, boxes_weight, boxes_num):
        num_proposals = int(boxes_num.data.cpu().numpy())
        hidden_encoder = self.initHidden()

        # encoder
        encoder_box_feature = F.relu(self.appear_linear(boxes_feature[0, :num_proposals, :]))

        encoder_box_score = F.relu(self.score_linear_2(self.score_linear_1(boxes_box_score[0, :num_proposals, :])))
        encoder_box_box = F.relu(self.box_linear(boxes_box[0, :num_proposals, :]))

        encoder_all_feature = torch.cat((encoder_box_feature, encoder_box_score, encoder_box_box), 1)
        encoder_all_feature_1 = F.relu(self.all_feature_linear(encoder_all_feature))
        encoder_input = torch.unsqueeze(encoder_all_feature_1, 1)
        # print(output.size())
        for j in range(self.n_layers):
            encoder_output, hidden_encoder = self.encoder_rnn(encoder_input, hidden_encoder)
            hidden_decoder = hidden_encoder
            decoder_output_list = []
            for index in range(encoder_input.size(0)):
                attn_weights = []
                for qq in range(encoder_input.size(0)):
                    aa = torch.cat((hidden_decoder.view(1, -1), encoder_input[qq, :, :].view(1, -1)), 1)
                    attn_weights.append(F.softmax(self.attn(aa), dim=1))

                attn_weights_all = torch.cat([bb for bb in attn_weights], 1)
                attn_applied = torch.bmm(attn_weights_all.unsqueeze(0), encoder_output.view(1,-1,self.hidden_size))
                decoder_input = torch.cat((encoder_input[index, :, :], attn_applied[0]), 1)
                decoder_input = self.attn_combine(decoder_input).unsqueeze(0)
                decoder_input = F.relu(decoder_input)
                decoder_output, hidden_decoder = self.decoder_rnn(decoder_input, hidden_decoder)
                decoder_output_list.append(decoder_output)

        decoder_output_all = torch.cat(decoder_output_list)
        output = self.sigmoid(self.out(decoder_output_all))
        label = boxes_label[0, :num_proposals]
        weight = boxes_weight[0, :num_proposals]
        # print(output)
        return output, label, weight

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

