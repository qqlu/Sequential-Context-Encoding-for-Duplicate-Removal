import torch, cv2
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import os.path as osp
# sys.path.append('/mnt/lustre/liushu1/qilu_ex/LSTM/')
from utils import stage2_assign_weight_slack_has_class
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

        proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_weight, proposals_class, proposals_keep_np = stage2_assign_weight_slack_has_class(box_feature, box_box, box_score, gts_box, weight=self.weight)

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
        box_class = proposals_class[sort_index, :]
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
        
        #class_split, equals all after nms result
        col_index = np.where(proposals_keep_np==1)[1]
        num_class_box = len(col_index)
        all_class_box_score = np.zeros((num_class_box, 32))
        all_class_box_origin_score = np.zeros((num_class_box, 1))
        all_class_box_box = np.zeros((num_class_box, 4))
        all_class_box_origin_box = np.zeros((num_class_box, 4))
        all_class_box_feature = np.zeros((num_class_box, 1024))
        all_class_box_label = np.zeros((num_class_box, 1))
        all_class_box_weight = np.ones((num_class_box, 1))
        all_class_box_class = np.zeros((num_class_box, 1))

        unique_class = np.zeros(80).astype(np.float)
        unique_class_len = np.zeros(81).astype(np.float)    # cumulative sum
        for ii in range(80):
            if ii in list(np.unique(col_index)):
                unique_class[ii] = 1
            unique_class_len[ii+1] = len(np.where(proposals_keep_np[:, ii]==1)[0]) + unique_class_len[ii]

        for ii in range(80):
            if unique_class[ii] == 0:
                continue
            start = int(unique_class_len[ii])
            end = int(unique_class_len[ii+1])
            class_split_index = np.where(box_keep_np[:, ii]==1)[0]
            class_split_sort_index = box_score_origin[class_split_index, ii].argsort()[::-1]
            class_sort_index = list(class_split_index[class_split_sort_index])

            all_class_box_feature[start:end,:] = box_feature[class_sort_index, :]
            all_class_box_label[start:end,:] = box_label[class_sort_index]
            all_class_box_box[start:end, 0:4] = box_box[class_sort_index, ii*4:(ii+1)*4]
            all_class_box_score[start:end, 0:32] = rank_score[class_sort_index, ii*32:(ii+1)*32]
            all_class_box_class[start:end, :] = int(ii)
            all_class_box_origin_score[start:end, 0] = box_score_origin[class_sort_index, ii]
            all_class_box_origin_box[start:end, 0:4] = box_box_origin[class_sort_index, ii*4:(ii+1)*4]
        
        index = np.where(all_class_box_label==1)[0]
        all_class_box_weight[index, :] = self.weight

        # print(all_class_box_box.shape)
        all_class_box_box = np.tile(all_class_box_box, 8)
        all_class_box_origin_score = np.tile(all_class_box_origin_score, 32)

        phase_np = np.zeros(1)
        if self.phase == 'train':
            phase_np[0] = 1
        if self.phase == 'bug':
            phase_np[0] = 2
        # class_split_score_feature or box_feature
        
        if self.phase == 'train':
            # print('dataset:', box_label)
            return box_feature, rank_score, box_box, box_label, box_weight, unique_class, unique_class_len, all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, phase_np
        elif self.phase == 'test':
            return box_feature, rank_score, box_box, all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_class, all_class_box_origin_score, all_class_box_origin_box, unique_class, unique_class_len, img_id, phase_np
        else:
            return box_score_origin, box_label, box_class, box_keep_np, box_weight, unique_class, unique_class_len, all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, phase_np

def unique_collate(batch):
    #print len(batch)
    if batch[0][-1][0]==1 or batch[0][-1][0]==2:
        return [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0])-1)]
    else:
        aa = [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0])-2)]
        aa.extend(torch.IntTensor(batch[0][-2]))
        return aa
        # return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3]), torch.FloatTensor(batch[0][4]), torch.FloatTensor(batch[0][5]), torch.IntTensor([batch[0][6]]), torch.FloatTensor(batch[0][7])

if __name__ == '__main__':
    base_path = '/mnt/lustre/liushu1/qilu_ex/dataset/coco/pytorch_data/'
    img_list = 'train.txt'
    use_mode = 'unique'
    cls_list = ['_' for _ in range(81)]
    train = TrainDataset(base_path, img_list, use_mode, cls_list, weight=2, phase='bug')
    num = len(train)
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    for i in range(num):
       box_score_origin, box_label, box_class, box_keep_np, box_weight, unique_class, unique_class_len, all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, phase_np = train[i]
       all_class_box_class = np.zeros((all_class_box_label.shape[0], 1)).astype(np.int)
       for ii in range(80):
            start = int(unique_class_len[ii])
            end =  int(unique_class_len[ii+1])
            all_class_box_class[start:end, :] = ii
       all_class_box_label = all_class_box_label.astype(np.int)
       box_class = box_class.astype(np.int)
       all_class_box_info = np.concatenate((all_class_box_class, all_class_box_label, all_class_box_origin_score), axis=1)
       # print('all_class_box_info:')
       # print(all_class_box_info)
       # # row_index = np.where(box_label==1)[0]
       # # col_index = box_class[row_index] - 1
       # # print(col_index)
       # # print(box_score_origin[row_index, col_index])
       # input()