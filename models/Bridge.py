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
from utils.bbox_ops import stage_full_assign_weight_slack_has_class_v6


class Bridge(nn.Module):
    def __init__(self, threshold=0.01, weight=2):
        super(Bridge, self).__init__()
        self.threshold = threshold
        self.search_table = self.rank_table()
        self.weight = weight

    def forward(self, pre_stage_output, pre_stage_last_feature, pre_stage_box_feature_variable, pre_stage_box_box_variable, pre_stage_box_score_variable, pre_stage_box_origin_score_variable, pre_stage_box_origin_box_variable, pre_stage_unique_class, pre_stage_unique_class_len, gts_box):
        post_stage_input = []
        save_score_final = pre_stage_output.data.view(-1, 1) * pre_stage_box_origin_score_variable[:, 0:1].data

        unique_class_len_sta = np.zeros(80)
        for ii in range(80):
            if(pre_stage_unique_class[ii]==0):
                continue
            start = int(pre_stage_unique_class_len[ii])
            end = int(pre_stage_unique_class_len[ii+1])
            final_valid = torch.nonzero(save_score_final[start:end, :]>self.threshold)
            if final_valid.numel()==0:
                continue
            final_valid = final_valid[:, 0]
            filter_num = final_valid.size(0)
            unique_class_len_sta[ii] = filter_num
        
        # initialization
        all_class_num = int(np.sum(unique_class_len_sta))
        post_stage_unique_class = torch.zeros(80)
        post_stage_unique_class_len = torch.zeros(81)

        post_stage_box_weight = np.ones((all_class_num, 1))
        post_stage_box_label = np.zeros((all_class_num, 1))

        post_stage_box_feature = torch.zeros(all_class_num, 1024)
        post_stage_box_box = torch.zeros(all_class_num, 32)
        post_stage_box_origin_box = torch.zeros(all_class_num, 4)
        # rank score is numpy type
        post_stage_box_score = np.zeros((all_class_num, 32))
        post_stage_box_origin_score = torch.zeros(all_class_num, 32)
        
        aa = []
        for ii in range(80):
            if pre_stage_unique_class[ii]==0:
                post_stage_unique_class[ii] = 0
                post_stage_unique_class_len[ii+1] = post_stage_unique_class_len[ii]
                continue
            
            start = int(pre_stage_unique_class_len[ii])
            end = int(pre_stage_unique_class_len[ii+1])
            final_valid = torch.nonzero(save_score_final[start:end, :]>self.threshold)
            if final_valid.numel()==0:
                post_stage_unique_class[ii] = 0
                post_stage_unique_class_len[ii+1] = post_stage_unique_class_len[ii]
                continue

            final_valid = final_valid[:, 0]
            filter_num = final_valid.size(0)
            
            final_valid_score = save_score_final[final_valid, :]
            _, final_sorted_index = torch.sort(final_valid_score, dim=0, descending=True)         
            # final_sort = final_valid[final_sorted_index.view(-1)].view(-1)
            final_sort = final_valid

            post_stage_unique_class[ii] = 1
            post_stage_unique_class_len[ii+1] = post_stage_unique_class_len[ii] + filter_num
            
            start_new = int(post_stage_unique_class_len[ii])
            end_new = int(post_stage_unique_class_len[ii+1])
            post_stage_box_feature[start_new:end_new, :] = pre_stage_box_feature_variable.data[start:end, :][final_sort, :]
            post_stage_box_box[start_new:end_new, :] = pre_stage_box_box_variable.data[start:end, :][final_sort, :]
            post_stage_box_origin_box[start_new:end_new, :] = pre_stage_box_origin_box_variable.data[start:end, :][final_sort, :]
            # print(post_stage_box_origin_score[start_new:end_new, :].size())
            # print(save_score_final[start:end, :][final_sort, :].size())
            post_stage_box_origin_score[start_new:end_new, :] = save_score_final[start:end, :][final_sort, :].expand(end_new-start_new, 32)
            post_stage_box_score[start_new:end_new, :] = self.search_table[0:filter_num, 0:32]
            post_stage_box_label[start_new:end_new, 0:1], post_stage_box_weight[start_new:end_new, 0:1] = stage_full_assign_weight_slack_has_class_v6(post_stage_box_origin_box[start_new:end_new, :].cpu().numpy(), post_stage_box_origin_score[start_new:end_new, 0:1].cpu().numpy(), gts_box.cpu().numpy(), ii, weight=self.weight)
            # post_stage_pre_last_feature[start_new:end_new, :] = pre_stage_last_feature[start:end, :][final_sort, :]
            aa.append(pre_stage_last_feature[start:end, :][final_sort, :])
        
        post_stage_pre_last_feature = torch.cat(aa)

        post_stage_box_weight_tensor = torch.FloatTensor(post_stage_box_weight).cuda()
        post_stage_box_label_tensor = Variable(torch.FloatTensor(post_stage_box_label).cuda())
        post_stage_box_feature_variable = Variable(post_stage_box_feature.cuda())
        post_stage_box_box_variaible = Variable(post_stage_box_box.cuda())
        post_stage_box_origin_box_tensor = post_stage_box_origin_box.cuda()
        post_stage_box_origin_score_variable = Variable(post_stage_box_origin_score.cuda())
        post_stage_box_score_variable = Variable(torch.FloatTensor(post_stage_box_score).cuda())
        
        post_stage_box_feature_variable.requires_grad = False
        post_stage_box_box_variaible.requires_grad = False
        post_stage_box_origin_score_variable.requires_grad = True
        post_stage_box_score_variable.requires_grad = False
        post_stage_input = [post_stage_box_weight_tensor, post_stage_box_label_tensor, post_stage_box_feature_variable, post_stage_box_box_variaible, post_stage_box_origin_box_tensor, post_stage_box_origin_score_variable, post_stage_box_score_variable, post_stage_unique_class, post_stage_unique_class_len]
        return post_stage_input, post_stage_pre_last_feature

    def rank_table(self):
        table = np.zeros((2500, 32), dtype=np.float)
        for i in range(2500):
            for j in range(16):
                table[i, 2*j:2*j+2] = i / (10000 ** (2*j/32.0))
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return table