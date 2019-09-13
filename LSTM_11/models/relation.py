from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from models.GlobalAttention import GlobalAttention

class Relation(nn.Module):
    def __init__(self, N_r, appear_dim=128):
        super(Relation, self).__init__()
        self.appear_dim = appear_dim
        self.d_k = 64
        self.d_g = 64
        self.N_r = 16
        # 128/16=8
        self.single_dim = self.appear_dim / self.N_r 
        
        # encoder
        self.W_V = nn.Linear(128, self.single_dim)
        self.W_K = nn.linear(128, self.d_k)
        self.W_Q = nn.linear(128, self.d_k)
        self.W_G = nn.linear(self.d_g, 1)


    def forward(self, appearance_feature, geometric_feature):
        appear = self.W_V(appearance_feature)
        geometric = geometric_feature.view(-1, self.d_g)

        w_A = torch.bmm(self.W_K(appearance_feature), self.W_Q(appearance_feature).view(self.d_k, 1)) / math.sqrt(self.d_k)
        w_G = F.relu(self.W_G(geometric)).view(geometric_feature.size(0), geometric_feature.size(1))

        w_nominator = w_G * torch.exp(w_A)
        w_denominator = torch.sum(w_nominator, 0).view(1, -1).expand_as(w_nominator)
        w =  w_nominator / (w_denominator)
        f_R = torch.bmm(w, appear)
        return f_R

class Multi_Relation(nn.Module):
    def __init__(self, N_r, appear_dim=128):
        super(Multi_Relation, self).__init__()
        self.appear_dim = appear_dim
        self.N_r = N_r
        # d_r = 128 / 16 = = 8
        self.d_r = self.appear_dim / self.N_r
        self.relation_list = nn.ModuleList([Relation(self.N_r, self.appear_dim) for i in range(self.N_r)])
        self.W_S = self.linear(self.appear_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, appearance_feature, geometric_feature, s_0):
        relation_feature = []
        for i in range(self.N_r):
            relation_feature.append(self.relation_list[i](appearance_feature, geometric_feature))

        all_relation = torch.cat(relation_feature, aixs=1)
        second = all_relation + appearance_feature
        s_1 = self.sigmoid(self.W_S(second))
        score = s_0 * s_1
        return score




