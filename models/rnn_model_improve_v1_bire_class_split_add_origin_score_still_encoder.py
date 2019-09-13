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
        # encoder
        self.encoder_rnn_0 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_1 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_2 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_3 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_4 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_5 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_6 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_7 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_8 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_9 = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        self.encoder_rnn_10 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_11 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_12 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_13 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_14 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_15 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_16 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_17 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_18 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_19 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.encoder_rnn_20 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_21 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_22 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_23 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_24 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_25 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_26 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_27 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_28 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_29 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.encoder_rnn_30 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_31 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_32 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_33 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_34 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_35 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_36 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_37 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_38 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_39 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.encoder_rnn_40 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_41 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_42 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_43 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_44 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_45 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_46 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_47 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_48 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_49 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.encoder_rnn_50 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_51 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_52 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_53 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_54 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_55 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_56 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_57 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_58 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_59 = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        self.encoder_rnn_60 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_61 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_62 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_63 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_64 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_65 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_66 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_67 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_68 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_69 = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        self.encoder_rnn_70 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_71 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_72 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_73 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_74 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_75 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_76 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_77 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_78 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.encoder_rnn_79 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        #decoder
        self.decoder_rnn_0 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_1 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_2 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_3 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_4 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_5 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_6 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_7 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_8 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_9 = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        self.decoder_rnn_10 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_11 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_12 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_13 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_14 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_15 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_16 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_17 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_18 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_19 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.decoder_rnn_20 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_21 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_22 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_23 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_24 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_25 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_26 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_27 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_28 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_29 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.decoder_rnn_30 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_31 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_32 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_33 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_34 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_35 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_36 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_37 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_38 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_39 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.decoder_rnn_40 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_41 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_42 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_43 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_44 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_45 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_46 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_47 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_48 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_49 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        self.decoder_rnn_50 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_51 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_52 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_53 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_54 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_55 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_56 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_57 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_58 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_59 = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        self.decoder_rnn_60 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_61 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_62 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_63 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_64 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_65 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_66 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_67 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_68 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_69 = nn.GRU(hidden_size, hidden_size, bidirectional=True)

        self.decoder_rnn_70 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_71 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_72 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_73 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_74 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_75 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_76 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_77 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_78 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.decoder_rnn_79 = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        
        # encoder shared weights
        self.appear_linear = nn.Linear(1024, hidden_size)

        # self.encoder_feature_linear = nn.Linear(3*hidden_size, hidden_size)
        self.feature_linear = nn.Linear(hidden_size+96, hidden_size)
        
        # decoder output
        self.out = nn.Linear(2*hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, boxes_feature, boxes_score, boxes_box, all_class_boxes_feature, all_class_boxes_score, all_class_boxes_box, all_class_boxes_origin_score, unique_class, unique_class_len):
        hidden_encoder_all = self.initHidden()
        # hidden_decoder = [0] * 80

        # encoder
        # encoder_box_feature = F.relu(self.appear_linear(boxes_feature))
        box_feature = F.relu(self.appear_linear(all_class_boxes_feature))

        # encoder_box_score = F.relu(self.score_linear_2(F.relu(self.score_linear_1(boxes_score))))
        # encoder_box_box = F.relu(self.box_linear(boxes_box))

        # encoder_all_feature = encoder_box_score

        # encoder_all_feature = torch.cat((encoder_box_feature, encoder_box_score, encoder_box_box), 1)
        all_feature = torch.cat((box_feature, all_class_boxes_score, all_class_boxes_box, all_class_boxes_origin_score), 1)

        # encoder_all_feature_1 = F.relu(self.encoder_feature_linear(encoder_all_feature))
        
        all_feature_1 = F.relu(self.feature_linear(all_feature))
        
        input_ = torch.unsqueeze(all_feature_1, 1)
        # decoder_input = torch.unsqueeze(decoder_all_feature_1, 1)

        # print(output.size())
        # for j in range(self.n_layers):
        #     # decoder_input = F.relu(encoder_input)
        #     encoder_output, hidden_encoder = self.encoder_rnn(encoder_input, hidden_encoder)
        
        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            class_encoder_input = input_[start:end, :, :]     
            if j==0:
                class_encoder_output, hidden_encoder_0 = self.encoder_rnn_0(class_encoder_input, hidden_encoder_all)
            elif j==1:
                class_encoder_output, hidden_encoder_1 = self.encoder_rnn_1(class_encoder_input, hidden_encoder_all)
            elif j==2:
                class_encoder_output, hidden_encoder_2 = self.encoder_rnn_2(class_encoder_input, hidden_encoder_all)
            elif j==3:
                class_encoder_output, hidden_encoder_3 = self.encoder_rnn_3(class_encoder_input, hidden_encoder_all)
            elif j==4:
                class_encoder_output, hidden_encoder_4 = self.encoder_rnn_4(class_encoder_input, hidden_encoder_all)
            elif j==5:
                class_encoder_output, hidden_encoder_5 = self.encoder_rnn_5(class_encoder_input, hidden_encoder_all)
            elif j==6:
                class_encoder_output, hidden_encoder_6 = self.encoder_rnn_6(class_encoder_input, hidden_encoder_all)
            elif j==7:
                class_encoder_output, hidden_encoder_7 = self.encoder_rnn_7(class_encoder_input, hidden_encoder_all)
            elif j==8:
                class_encoder_output, hidden_encoder_8 = self.encoder_rnn_8(class_encoder_input, hidden_encoder_all)
            elif j==9:
                class_encoder_output, hidden_encoder_9 = self.encoder_rnn_9(class_encoder_input, hidden_encoder_all)
            elif j==10:
                class_encoder_output, hidden_encoder_10 = self.encoder_rnn_10(class_encoder_input, hidden_encoder_all)
            elif j==11:
                class_encoder_output, hidden_encoder_11 = self.encoder_rnn_11(class_encoder_input, hidden_encoder_all)
            elif j==12:
                class_encoder_output, hidden_encoder_12 = self.encoder_rnn_12(class_encoder_input, hidden_encoder_all)
            elif j==13:
                class_encoder_output, hidden_encoder_13 = self.encoder_rnn_13(class_encoder_input, hidden_encoder_all)
            elif j==14:
                class_encoder_output, hidden_encoder_14 = self.encoder_rnn_14(class_encoder_input, hidden_encoder_all)
            elif j==15:
                class_encoder_output, hidden_encoder_15 = self.encoder_rnn_15(class_encoder_input, hidden_encoder_all)
            elif j==16:
                class_encoder_output, hidden_encoder_16 = self.encoder_rnn_16(class_encoder_input, hidden_encoder_all)
            elif j==17:
                class_encoder_output, hidden_encoder_17 = self.encoder_rnn_17(class_encoder_input, hidden_encoder_all)
            elif j==18:
                class_encoder_output, hidden_encoder_18 = self.encoder_rnn_18(class_encoder_input, hidden_encoder_all)
            elif j==19:
                class_encoder_output, hidden_encoder_19 = self.encoder_rnn_19(class_encoder_input, hidden_encoder_all)
            elif j==20:
                class_encoder_output, hidden_encoder_20 = self.encoder_rnn_20(class_encoder_input, hidden_encoder_all)
            elif j==21:
                class_encoder_output, hidden_encoder_21 = self.encoder_rnn_21(class_encoder_input, hidden_encoder_all)
            elif j==22:
                class_encoder_output, hidden_encoder_22 = self.encoder_rnn_22(class_encoder_input, hidden_encoder_all)
            elif j==23:
                class_encoder_output, hidden_encoder_23 = self.encoder_rnn_23(class_encoder_input, hidden_encoder_all)
            elif j==24:
                class_encoder_output, hidden_encoder_24 = self.encoder_rnn_24(class_encoder_input, hidden_encoder_all)
            elif j==25:
                class_encoder_output, hidden_encoder_25 = self.encoder_rnn_25(class_encoder_input, hidden_encoder_all)
            elif j==26:
                class_encoder_output, hidden_encoder_26 = self.encoder_rnn_26(class_encoder_input, hidden_encoder_all)
            elif j==27:
                class_encoder_output, hidden_encoder_27 = self.encoder_rnn_27(class_encoder_input, hidden_encoder_all)
            elif j==28:
                class_encoder_output, hidden_encoder_28 = self.encoder_rnn_28(class_encoder_input, hidden_encoder_all)
            elif j==29:
                class_encoder_output, hidden_encoder_29 = self.encoder_rnn_29(class_encoder_input, hidden_encoder_all)
            elif j==30:
                class_encoder_output, hidden_encoder_30 = self.encoder_rnn_30(class_encoder_input, hidden_encoder_all)
            elif j==31:
                class_encoder_output, hidden_encoder_31 = self.encoder_rnn_31(class_encoder_input, hidden_encoder_all)
            elif j==32:
                class_encoder_output, hidden_encoder_32 = self.encoder_rnn_32(class_encoder_input, hidden_encoder_all)
            elif j==33:
                class_encoder_output, hidden_encoder_33 = self.encoder_rnn_33(class_encoder_input, hidden_encoder_all)
            elif j==34:
                class_encoder_output, hidden_encoder_34 = self.encoder_rnn_34(class_encoder_input, hidden_encoder_all)
            elif j==35:
                class_encoder_output, hidden_encoder_35 = self.encoder_rnn_35(class_encoder_input, hidden_encoder_all)
            elif j==36:
                class_encoder_output, hidden_encoder_36 = self.encoder_rnn_36(class_encoder_input, hidden_encoder_all)
            elif j==37:
                class_encoder_output, hidden_encoder_37 = self.encoder_rnn_37(class_encoder_input, hidden_encoder_all)
            elif j==38:
                class_encoder_output, hidden_encoder_38 = self.encoder_rnn_38(class_encoder_input, hidden_encoder_all)
            elif j==39:
                class_encoder_output, hidden_encoder_39 = self.encoder_rnn_39(class_encoder_input, hidden_encoder_all)
            elif j==40:
                class_encoder_output, hidden_encoder_40 = self.encoder_rnn_40(class_encoder_input, hidden_encoder_all)
            elif j==41:
                class_encoder_output, hidden_encoder_41 = self.encoder_rnn_41(class_encoder_input, hidden_encoder_all)
            elif j==42:
                class_encoder_output, hidden_encoder_42 = self.encoder_rnn_42(class_encoder_input, hidden_encoder_all)
            elif j==43:
                class_encoder_output, hidden_encoder_43 = self.encoder_rnn_43(class_encoder_input, hidden_encoder_all)
            elif j==44:
                class_encoder_output, hidden_encoder_44 = self.encoder_rnn_44(class_encoder_input, hidden_encoder_all)
            elif j==45:
                class_encoder_output, hidden_encoder_45 = self.encoder_rnn_45(class_encoder_input, hidden_encoder_all)
            elif j==46:
                class_encoder_output, hidden_encoder_46 = self.encoder_rnn_46(class_encoder_input, hidden_encoder_all)
            elif j==47:
                class_encoder_output, hidden_encoder_47 = self.encoder_rnn_47(class_encoder_input, hidden_encoder_all)
            elif j==48:
                class_encoder_output, hidden_encoder_48 = self.encoder_rnn_48(class_encoder_input, hidden_encoder_all)
            elif j==49:
                class_encoder_output, hidden_encoder_49 = self.encoder_rnn_49(class_encoder_input, hidden_encoder_all)
            elif j==50:
                class_encoder_output, hidden_encoder_50 = self.encoder_rnn_50(class_encoder_input, hidden_encoder_all)
            elif j==51:
                class_encoder_output, hidden_encoder_51 = self.encoder_rnn_51(class_encoder_input, hidden_encoder_all)
            elif j==52:
                class_encoder_output, hidden_encoder_52 = self.encoder_rnn_52(class_encoder_input, hidden_encoder_all)
            elif j==53:
                class_encoder_output, hidden_encoder_53 = self.encoder_rnn_53(class_encoder_input, hidden_encoder_all)
            elif j==54:
                class_encoder_output, hidden_encoder_54 = self.encoder_rnn_54(class_encoder_input, hidden_encoder_all)
            elif j==55:
                class_encoder_output, hidden_encoder_55 = self.encoder_rnn_55(class_encoder_input, hidden_encoder_all)
            elif j==56:
                class_encoder_output, hidden_encoder_56 = self.encoder_rnn_56(class_encoder_input, hidden_encoder_all)
            elif j==57:
                class_encoder_output, hidden_encoder_57 = self.encoder_rnn_57(class_encoder_input, hidden_encoder_all)
            elif j==58:
                class_encoder_output, hidden_encoder_58 = self.encoder_rnn_58(class_encoder_input, hidden_encoder_all)
            elif j==59:
                class_encoder_output, hidden_encoder_59 = self.encoder_rnn_59(class_encoder_input, hidden_encoder_all)
            elif j==60:
                class_encoder_output, hidden_encoder_60 = self.encoder_rnn_60(class_encoder_input, hidden_encoder_all)
            elif j==61:
                class_encoder_output, hidden_encoder_61 = self.encoder_rnn_61(class_encoder_input, hidden_encoder_all)
            elif j==62:
                class_encoder_output, hidden_encoder_62 = self.encoder_rnn_62(class_encoder_input, hidden_encoder_all)
            elif j==63:
                class_encoder_output, hidden_encoder_63 = self.encoder_rnn_63(class_encoder_input, hidden_encoder_all)
            elif j==64:
                class_encoder_output, hidden_encoder_64 = self.encoder_rnn_64(class_encoder_input, hidden_encoder_all)
            elif j==65:
                class_encoder_output, hidden_encoder_65 = self.encoder_rnn_65(class_encoder_input, hidden_encoder_all)
            elif j==66:
                class_encoder_output, hidden_encoder_66 = self.encoder_rnn_66(class_encoder_input, hidden_encoder_all)
            elif j==67:
                class_encoder_output, hidden_encoder_67 = self.encoder_rnn_67(class_encoder_input, hidden_encoder_all)
            elif j==68:
                class_encoder_output, hidden_encoder_68 = self.encoder_rnn_68(class_encoder_input, hidden_encoder_all)
            elif j==69:
                class_encoder_output, hidden_encoder_69 = self.encoder_rnn_69(class_encoder_input, hidden_encoder_all)
            elif j==70:
                class_encoder_output, hidden_encoder_70 = self.encoder_rnn_70(class_encoder_input, hidden_encoder_all)
            elif j==71:
                class_encoder_output, hidden_encoder_71 = self.encoder_rnn_71(class_encoder_input, hidden_encoder_all)
            elif j==72:
                class_encoder_output, hidden_encoder_72 = self.encoder_rnn_72(class_encoder_input, hidden_encoder_all)
            elif j==73:
                class_encoder_output, hidden_encoder_73 = self.encoder_rnn_73(class_encoder_input, hidden_encoder_all)
            elif j==74:
                class_encoder_output, hidden_encoder_74 = self.encoder_rnn_74(class_encoder_input, hidden_encoder_all)
            elif j==75:
                class_encoder_output, hidden_encoder_75 = self.encoder_rnn_75(class_encoder_input, hidden_encoder_all)
            elif j==76:
                class_encoder_output, hidden_encoder_76 = self.encoder_rnn_76(class_encoder_input, hidden_encoder_all)
            elif j==77:
                class_encoder_output, hidden_encoder_77 = self.encoder_rnn_77(class_encoder_input, hidden_encoder_all)
            elif j==78:
                class_encoder_output, hidden_encoder_78 = self.encoder_rnn_78(class_encoder_input, hidden_encoder_all)
            elif j==79:
                class_encoder_output, hidden_encoder_79 = self.encoder_rnn_79(class_encoder_input, hidden_encoder_all)

        decoder_output_list = []
        # hidden_decoder = hidden_encoder
        # decoder_rnn = self.decoder_rnn
        for j in range(80):
            if(unique_class[j]==0):
                continue
            start = int(unique_class_len[j])
            end = int(unique_class_len[j+1])
            class_decoder_input = input_[start:end, :, :]     
            if j==0:
                class_decoder_output, hidden_decoder = self.decoder_rnn_0(class_decoder_input, hidden_encoder_0)
            elif j==1:
                class_decoder_output, hidden_decoder = self.decoder_rnn_1(class_decoder_input, hidden_encoder_1)
            elif j==2:
                class_decoder_output, hidden_decoder = self.decoder_rnn_2(class_decoder_input, hidden_encoder_2)
            elif j==3:
                class_decoder_output, hidden_decoder = self.decoder_rnn_3(class_decoder_input, hidden_encoder_3)
            elif j==4:
                class_decoder_output, hidden_decoder = self.decoder_rnn_4(class_decoder_input, hidden_encoder_4)
            elif j==5:
                class_decoder_output, hidden_decoder = self.decoder_rnn_5(class_decoder_input, hidden_encoder_5)
            elif j==6:
                class_decoder_output, hidden_decoder = self.decoder_rnn_6(class_decoder_input, hidden_encoder_6)
            elif j==7:
                class_decoder_output, hidden_decoder = self.decoder_rnn_7(class_decoder_input, hidden_encoder_7)
            elif j==8:
                class_decoder_output, hidden_decoder = self.decoder_rnn_8(class_decoder_input, hidden_encoder_8)
            elif j==9:
                class_decoder_output, hidden_decoder = self.decoder_rnn_9(class_decoder_input, hidden_encoder_9)
            elif j==10:
                class_decoder_output, hidden_decoder = self.decoder_rnn_10(class_decoder_input, hidden_encoder_10)
            elif j==11:
                class_decoder_output, hidden_decoder = self.decoder_rnn_11(class_decoder_input, hidden_encoder_11)
            elif j==12:
                class_decoder_output, hidden_decoder = self.decoder_rnn_12(class_decoder_input, hidden_encoder_12)
            elif j==13:
                class_decoder_output, hidden_decoder = self.decoder_rnn_13(class_decoder_input, hidden_encoder_13)
            elif j==14:
                class_decoder_output, hidden_decoder = self.decoder_rnn_14(class_decoder_input, hidden_encoder_14)
            elif j==15:
                class_decoder_output, hidden_decoder = self.decoder_rnn_15(class_decoder_input, hidden_encoder_15)
            elif j==16:
                class_decoder_output, hidden_decoder = self.decoder_rnn_16(class_decoder_input, hidden_encoder_16)
            elif j==17:
                class_decoder_output, hidden_decoder = self.decoder_rnn_17(class_decoder_input, hidden_encoder_17)
            elif j==18:
                class_decoder_output, hidden_decoder = self.decoder_rnn_18(class_decoder_input, hidden_encoder_18)
            elif j==19:
                class_decoder_output, hidden_decoder = self.decoder_rnn_19(class_decoder_input, hidden_encoder_19)
            elif j==20:
                class_decoder_output, hidden_decoder = self.decoder_rnn_20(class_decoder_input, hidden_encoder_20)
            elif j==21:
                class_decoder_output, hidden_decoder = self.decoder_rnn_21(class_decoder_input, hidden_encoder_21)
            elif j==22:
                class_decoder_output, hidden_decoder = self.decoder_rnn_22(class_decoder_input, hidden_encoder_22)
            elif j==23:
                class_decoder_output, hidden_decoder = self.decoder_rnn_23(class_decoder_input, hidden_encoder_23)
            elif j==24:
                class_decoder_output, hidden_decoder = self.decoder_rnn_24(class_decoder_input, hidden_encoder_24)
            elif j==25:
                class_decoder_output, hidden_decoder = self.decoder_rnn_25(class_decoder_input, hidden_encoder_25)
            elif j==26:
                class_decoder_output, hidden_decoder = self.decoder_rnn_26(class_decoder_input, hidden_encoder_26)
            elif j==27:
                class_decoder_output, hidden_decoder = self.decoder_rnn_27(class_decoder_input, hidden_encoder_27)
            elif j==28:
                class_decoder_output, hidden_decoder = self.decoder_rnn_28(class_decoder_input, hidden_encoder_28)
            elif j==29:
                class_decoder_output, hidden_decoder = self.decoder_rnn_29(class_decoder_input, hidden_encoder_29)
            elif j==30:
                class_decoder_output, hidden_decoder = self.decoder_rnn_30(class_decoder_input, hidden_encoder_30)
            elif j==31:
                class_decoder_output, hidden_decoder = self.decoder_rnn_31(class_decoder_input, hidden_encoder_31)
            elif j==32:
                class_decoder_output, hidden_decoder = self.decoder_rnn_32(class_decoder_input, hidden_encoder_32)
            elif j==33:
                class_decoder_output, hidden_decoder = self.decoder_rnn_33(class_decoder_input, hidden_encoder_33)
            elif j==34:
                class_decoder_output, hidden_decoder = self.decoder_rnn_34(class_decoder_input, hidden_encoder_34)
            elif j==35:
                class_decoder_output, hidden_decoder = self.decoder_rnn_35(class_decoder_input, hidden_encoder_35)
            elif j==36:
                class_decoder_output, hidden_decoder = self.decoder_rnn_36(class_decoder_input, hidden_encoder_36)
            elif j==37:
                class_decoder_output, hidden_decoder = self.decoder_rnn_37(class_decoder_input, hidden_encoder_37)
            elif j==38:
                class_decoder_output, hidden_decoder = self.decoder_rnn_38(class_decoder_input, hidden_encoder_38)
            elif j==39:
                class_decoder_output, hidden_decoder = self.decoder_rnn_39(class_decoder_input, hidden_encoder_39)
            elif j==40:
                class_decoder_output, hidden_decoder = self.decoder_rnn_40(class_decoder_input, hidden_encoder_40)
            elif j==41:
                class_decoder_output, hidden_decoder = self.decoder_rnn_41(class_decoder_input, hidden_encoder_41)
            elif j==42:
                class_decoder_output, hidden_decoder = self.decoder_rnn_42(class_decoder_input, hidden_encoder_42)
            elif j==43:
                class_decoder_output, hidden_decoder = self.decoder_rnn_43(class_decoder_input, hidden_encoder_43)
            elif j==44:
                class_decoder_output, hidden_decoder = self.decoder_rnn_44(class_decoder_input, hidden_encoder_44)
            elif j==45:
                class_decoder_output, hidden_decoder = self.decoder_rnn_45(class_decoder_input, hidden_encoder_45)
            elif j==46:
                class_decoder_output, hidden_decoder = self.decoder_rnn_46(class_decoder_input, hidden_encoder_46)
            elif j==47:
                class_decoder_output, hidden_decoder = self.decoder_rnn_47(class_decoder_input, hidden_encoder_47)
            elif j==48:
                class_decoder_output, hidden_decoder = self.decoder_rnn_48(class_decoder_input, hidden_encoder_48)
            elif j==49:
                class_decoder_output, hidden_decoder = self.decoder_rnn_49(class_decoder_input, hidden_encoder_49)
            elif j==50:
                class_decoder_output, hidden_decoder = self.decoder_rnn_50(class_decoder_input, hidden_encoder_50)
            elif j==51:
                class_decoder_output, hidden_decoder = self.decoder_rnn_51(class_decoder_input, hidden_encoder_51)
            elif j==52:
                class_decoder_output, hidden_decoder = self.decoder_rnn_52(class_decoder_input, hidden_encoder_52)
            elif j==53:
                class_decoder_output, hidden_decoder = self.decoder_rnn_53(class_decoder_input, hidden_encoder_53)
            elif j==54:
                class_decoder_output, hidden_decoder = self.decoder_rnn_54(class_decoder_input, hidden_encoder_54)
            elif j==55:
                class_decoder_output, hidden_decoder = self.decoder_rnn_55(class_decoder_input, hidden_encoder_55)
            elif j==56:
                class_decoder_output, hidden_decoder = self.decoder_rnn_56(class_decoder_input, hidden_encoder_56)
            elif j==57:
                class_decoder_output, hidden_decoder = self.decoder_rnn_57(class_decoder_input, hidden_encoder_57)
            elif j==58:
                class_decoder_output, hidden_decoder = self.decoder_rnn_58(class_decoder_input, hidden_encoder_58)
            elif j==59:
                class_decoder_output, hidden_decoder = self.decoder_rnn_59(class_decoder_input, hidden_encoder_59)
            elif j==60:
                class_decoder_output, hidden_decoder = self.decoder_rnn_60(class_decoder_input, hidden_encoder_60)
            elif j==61:
                class_decoder_output, hidden_decoder = self.decoder_rnn_61(class_decoder_input, hidden_encoder_61)
            elif j==62:
                class_decoder_output, hidden_decoder = self.decoder_rnn_62(class_decoder_input, hidden_encoder_62)
            elif j==63:
                class_decoder_output, hidden_decoder = self.decoder_rnn_63(class_decoder_input, hidden_encoder_63)
            elif j==64:
                class_decoder_output, hidden_decoder = self.decoder_rnn_64(class_decoder_input, hidden_encoder_64)
            elif j==65:
                class_decoder_output, hidden_decoder = self.decoder_rnn_65(class_decoder_input, hidden_encoder_65)
            elif j==66:
                class_decoder_output, hidden_decoder = self.decoder_rnn_66(class_decoder_input, hidden_encoder_66)
            elif j==67:
                class_decoder_output, hidden_decoder = self.decoder_rnn_67(class_decoder_input, hidden_encoder_67)
            elif j==68:
                class_decoder_output, hidden_decoder = self.decoder_rnn_68(class_decoder_input, hidden_encoder_68)
            elif j==69:
                class_decoder_output, hidden_decoder = self.decoder_rnn_69(class_decoder_input, hidden_encoder_69)
            elif j==70:
                class_decoder_output, hidden_decoder = self.decoder_rnn_70(class_decoder_input, hidden_encoder_70)
            elif j==71:
                class_decoder_output, hidden_decoder = self.decoder_rnn_71(class_decoder_input, hidden_encoder_71)
            elif j==72:
                class_decoder_output, hidden_decoder = self.decoder_rnn_72(class_decoder_input, hidden_encoder_72)
            elif j==73:
                class_decoder_output, hidden_decoder = self.decoder_rnn_73(class_decoder_input, hidden_encoder_73)
            elif j==74:
                class_decoder_output, hidden_decoder = self.decoder_rnn_74(class_decoder_input, hidden_encoder_74)
            elif j==75:
                class_decoder_output, hidden_decoder = self.decoder_rnn_75(class_decoder_input, hidden_encoder_75)
            elif j==76:
                class_decoder_output, hidden_decoder = self.decoder_rnn_76(class_decoder_input, hidden_encoder_76)
            elif j==77:
                class_decoder_output, hidden_decoder = self.decoder_rnn_77(class_decoder_input, hidden_encoder_77)
            elif j==78:
                class_decoder_output, hidden_decoder = self.decoder_rnn_78(class_decoder_input, hidden_encoder_78)
            elif j==79:
                class_decoder_output, hidden_decoder = self.decoder_rnn_79(class_decoder_input, hidden_encoder_79)
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