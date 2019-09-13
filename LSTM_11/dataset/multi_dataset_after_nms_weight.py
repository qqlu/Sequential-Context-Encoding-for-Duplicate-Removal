import torch, cv2
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import os.path as osp
from utils import stage2_assign_weight_slack
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
        self.info_path = os.path.join(self.base_path, 'info_1/')
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
    def __init__(self, base_path, txt_name, use_mode, cls_list, overlap_thresh=0.5, score_thresh=0.01, weight=1.2, phase='train'):
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

        proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_weight, proposals_keep_np = stage2_assign_weight_slack(box_feature, box_box, box_score, gts_box, weight=self.weight, nms_iou_thr=0.5)

        box_sort_score = proposals_score_nms[:, 1:]
        box_feature = proposals_feature_nms
        box_box = proposals_box_nms[:, 4:]

        box_sort_score_max = np.max(box_sort_score, axis=1)
        # logger1.info(box_sort_score_max.reshape(-1).shape)
        sort_index = box_sort_score_max.argsort()[::-1]
        box_score = box_sort_score[sort_index, :]
        box_feature = box_feature[sort_index, :]
        box_box = box_box[sort_index, :]
        box_label = proposals_label[sort_index, :]
        box_weight = proposals_weight[sort_index, :]
        box_keep_np = proposals_keep_np[sort_index, :]
        # origin box_box and box_score
        box_box_origin = box_box.copy()
        box_score_origin = box_score.copy()
        
        box_box[:, 0::2] = np.log(box_box[:, 0::2] / float(width) + 0.5)
        box_box[:, 1::2] = np.log(box_box[:, 1::2] / float(height) + 0.5)
                
        num_box = box_feature.shape[0]
        ranks = np.empty_like(box_score)
        for ii in range(box_score.shape[1]):
            temp = box_score[:, ii].argsort()
            ranks[temp, ii] = np.arange(box_score.shape[0], 0, -1)

        rank_score = np.zeros((num_box, 80*32))
        for ii in range(ranks.shape[0]):
            for jj in range(ranks.shape[1]): 
                rank_score[ii, jj*32: (jj+1)*32] = self.search_table[int(ranks[ii, jj]), :]
        
        if self.phase == 'train':
            # print('dataset:', box_label)
            return box_feature, rank_score, box_box, box_label, box_weight
        elif self.phase == 'test':
            return box_feature, rank_score, box_box, box_label, box_score_origin, box_box_origin, img_id, box_keep_np

def unique_collate(batch):
    #print len(batch)
    if len(batch[0]) == 3:
        return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2])
    elif len(batch[0]) == 4:
        return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3])
    elif len(batch[0]) == 5:
        return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3]), torch.FloatTensor(batch[0][4])
    elif len(batch[0]) == 8:
        # print('unique:', np.max(batch[0][4].reshape(-1)))
        return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3]), torch.FloatTensor(batch[0][4]), torch.FloatTensor(batch[0][5]), torch.IntTensor([batch[0][6]]), torch.FloatTensor(batch[0][7])

if __name__ == '__main__':
    base_path = '/data/dataset/LSTM_data/'