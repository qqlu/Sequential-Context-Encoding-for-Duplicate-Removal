import torch, cv2
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import os.path as osp
sys.path.append('/data/luqi/LSTM/')
from utils import stage_full_assign_weight_slack_has_class
import pickle as pkl
import cvbase as cvb
import time
import math
import time
import logging
from utils import solver_log

class MultiDataset(Dataset):
    def __init__(self, base_path, txt_name):
        super(MultiDataset, self).__init__()
        self.base_path = base_path
        self.txt_path = os.path.join(self.base_path, txt_name)
        self.image_index = cvb.list_from_file(self.txt_path)
        self.info_path = os.path.join(self.base_path, 'info/')
        self.gt_path = os.path.join(self.base_path, 'gt/')
        self.img_path = os.path.join(self.base_path, 'img/')
        self.search_table = self.rank_table()

    def __len__(self):
        return len(self.image_index)

    def rank_table(self):
        table = np.zeros((1500, 32), dtype=np.float)
        for i in range(1500):
            for j in range(16):
                table[i, 2*j:2*j+2] = i / (10000 ** (2*j/32.0))
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return table

class TrainDataset(MultiDataset):
    def __init__(self, base_path, txt_name, use_mode, cls_list, overlap_thresh=0.5, score_thresh=0.05, weight=1000, phase='train'):
        super(TrainDataset, self).__init__(base_path, txt_name)
        # The use_mode is unique or msra
        self.use_mode = use_mode
        self.cls_list = cls_list
        self.cls_num = len(cls_list)
        self.overlap_thresh = overlap_thresh
        self.score_thresh = score_thresh
        self.phase = phase
        self.weight = weight

    def __getitem__(self, idx):
        # logger1 = solver_log(os.path.join('/mnt/lustre/liushu1/', 'load_dataset.log'))
        info_pkl_path = os.path.join(self.info_path, self.image_index[idx] + '.pkl')
        img_id = int(float(self.image_index[idx]))
        img_path = os.path.join(self.img_path, self.image_index[idx] + '.jpg')
        img = cv2.imread(img_path)
        height, width = img.shape[0:2]
            
        box_feature, box_box, box_score = pkl.load(open(os.path.join(info_pkl_path), 'rb'), encoding='iso-8859-1')
        box_feature = box_feature.astype(np.float)
        box_box = box_box.astype(np.float)
        box_score = box_score.astype(np.float)

        gts_info = pkl.load(open(os.path.join(self.gt_path, self.image_index[idx] + '.pkl'), 'rb'), encoding='iso-8859-1')
        gts_box = np.zeros((len(gts_info), 5))
        for index, gt in enumerate(gts_info):
            gts_box[index, :] = gt['bbox']

        box_label, box_weight = stage_full_assign_weight_slack_has_class(box_feature, box_box, box_score, gts_box, weight=self.weight)

        box_score = box_score[:, 1:]
        # box_feature = proposals_feature
        box_box = box_box[:, 4:]

        box_score_max = np.max(box_score, axis=1)
        valid_index = np.where(box_score_max>=self.score_thresh)[0]
        box_feature = box_feature[valid_index, :]
        box_box = box_box[valid_index, :]
        box_score = box_score[valid_index, :]
        box_label = box_label[valid_index, :]
        box_weight = box_weight[valid_index, :]

        # origin box_box and box_score
        box_box_origin = box_box.copy()
        box_score_origin = box_score.copy()
        
        box_box[:, 0::2] = np.log(box_box[:, 0::2] / float(width) + 0.5)
        box_box[:, 1::2] = np.log(box_box[:, 1::2] / float(height) + 0.5)
                
        num_box = box_feature.shape[0]
        ranks = np.empty_like(box_score)
        all_class_box_weight = np.ones((num_box*80, 1))
        all_class_box_label = np.zeros((num_box*80, 1))
        all_class_box_box = np.zeros((80, num_box, 32))
        all_class_box_origin_score = np.zeros((80, num_box, 32))
        all_class_box_score = np.zeros((80, num_box, 32))
        for ii in range(80):
            temp = box_score[:, ii].argsort()[::-1]
            all_class_box_weight[ii*num_box:(ii+1)*num_box, 0] = box_weight[temp, ii]
            all_class_box_label[ii*num_box:(ii+1)*num_box, 0] = box_label[temp, ii]
            all_class_box_box[ii, :, :] = np.tile(box_box[temp, ii*4:(ii+1)*4], 8)
            all_class_box_origin_score[ii, :, :] = np.tile(box_score_origin[temp, ii:ii+1], 32)
            all_class_box_score[ii, :, :] = self.search_table[0:num_box, 0:32]
            ranks[:, ii] = np.arange(0, num_box, 1)
        # print(all_class_box_label.shape)
        # print(num_box)
        phase_np = np.zeros(1)
        if self.phase == 'train':
            phase_np[0] = 1
        if self.phase == 'bug':
            phase_np[0] = 2
        # class_split_score_feature or box_feature
        
        if self.phase == 'train':
            return box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, ranks, phase_np
        elif self.phase == 'test':
            return box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, ranks, img_id, phase_np
        

def unique_collate(batch):
    #print len(batch)
    if batch[0][-1][0]==1 or batch[0][-1][0]==2:
        # print(len(batch[0]))
        aa = [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0])-2)]
        aa.extend([torch.IntTensor(batch[0][-2])])
        # print(len(aa))
        # bb = [torch.IntTensor(batch[0][-2])]
        # aa.extend(bb)
        # aa.extend(bb)
        # print(len(aa))
        return aa
    else:
        aa = [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0])-2)]
        aa.extend(torch.IntTensor(batch[0][-2]))
        return aa
        # return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3]), torch.FloatTensor(batch[0][4]), torch.FloatTensor(batch[0][5]), torch.IntTensor([batch[0][6]]), torch.FloatTensor(batch[0][7])

if __name__ == '__main__':
    base_path = '/data/luqi/dataset/pytorch_data/'
    img_list = 'train.txt'
    use_mode = 'unique'
    cls_list = ['_' for _ in range(81)]
    train = TrainDataset(base_path, img_list, use_mode, cls_list, weight=2, phase='train')
    num = len(train)
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    for i in range(num):
       box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, ranks, phase_np = train[i]
       # all_class_box_class = np.zeros((all_class_box_label.shape[0], 1)).astype(np.int)
       # for ii in range(80):
       #      start = int(unique_class_len[ii])
       #      end =  int(unique_class_len[ii+1])
       #      all_class_box_class[start:end, :] = ii
       # all_class_box_label = all_class_box_label.astype(np.int)
       # box_class = box_class.astype(np.int)
       # all_class_box_info = np.concatenate((all_class_box_class, all_class_box_label, all_class_box_origin_score), axis=1)
       # print('all_class_box_info:')
       # print(all_class_box_info)
       # # row_index = np.where(box_label==1)[0]
       # # col_index = box_class[row_index] - 1
       # # print(col_index)
       # # print(box_score_origin[row_index, col_index])
       # print(box_feature.shape)
       # print(ranks[:,0].shape)
       # print(all_class_box_origin_score[0,:,0].shape)
       # print(np.concatenate((ranks[:, 0:1], all_class_box_origin_score[0, :, 0].reshape(-1, 1), all_class_box_label.reshape(80,-1,1)[0, :, 0].reshape(-1, 1), all_class_box_box[0, :, 0:4].reshape(-1, 4)), axis=1))
       # print(all_class_box_origin_score[:, 0])
       # input()