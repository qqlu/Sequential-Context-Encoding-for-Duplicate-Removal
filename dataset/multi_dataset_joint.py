import torch, cv2
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import os.path as osp
sys.path.append('/data/luqi/LSTM/')
# from utils.bbox_ops import stage1_before_assign_weight_slack_has_class
from utils.bbox_ops import stage1_before_no_gt_assign_weight_slack_has_class
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
        table = np.zeros((2500, 32), dtype=np.float)
        for i in range(2500):
            for j in range(16):
                table[i, 2*j:2*j+2] = i / (10000 ** (2*j/32.0))
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return table

class TrainDataset(MultiDataset):
    def __init__(self, base_path, txt_name, overlap_thresh=0.5, score_thresh=0.01, weight=4):
        super(TrainDataset, self).__init__(base_path, txt_name)
        self.overlap_thresh = overlap_thresh
        self.score_thresh = score_thresh
        self.weight = weight

    def __getitem__(self, idx):
        # logger1 = solver_log(os.path.join('/mnt/lustre/liushu1/', 'load_dataset.log'))
        info_pkl_path = os.path.join(self.info_path, self.image_index[idx] + '.pkl')
        img_id = np.zeros((1))
        img_id[0] = int(float(self.image_index[idx]))

        img_path = os.path.join(self.img_path, self.image_index[idx] + '.jpg')
        img = cv2.imread(img_path)
        height, width = img.shape[0:2]
            
        box_feature, box_box, box_score = pkl.load(open(os.path.join(info_pkl_path), 'rb'), encoding='iso-8859-1')
        box_feature = box_feature.astype(np.float)
        box_box = box_box.astype(np.float)
        box_score = box_score.astype(np.float)

        # gts_info = pkl.load(open(os.path.join(self.gt_path, self.image_index[idx] + '.pkl'), 'rb'), encoding='iso-8859-1')
        gts_info = pkl.load(open(os.path.join(self.gt_path, self.image_index[idx] + '.pkl'), 'rb'), encoding='iso-8859-1')
        gts_box = np.zeros((len(gts_info), 5))
        for index, gt in enumerate(gts_info):
            gts_box[index, :] = gt['bbox']

        # box_label, box_weight = stage1_before_assign_weight_slack_has_class(box_feature, box_box, box_score, gts_box, weight=self.weight)
        box_label, box_weight = stage1_before_no_gt_assign_weight_slack_has_class(box_feature, box_box, box_score, gts_box, weight=self.weight)

        box_score = box_score[:, 1:]
        # box_feature = proposals_feature
        box_box = box_box[:, 4:]

        unique_class = np.zeros(80).astype(np.float)
        unique_class_len = np.zeros(81).astype(np.float)    # cumulative sum
        cls_num = 0
        cls_list = []
        label_num = 0
        for ii in range(80):
            valid_index = np.where(box_score[:,ii]>=self.score_thresh)[0]
            cls_list.append(valid_index)
            valid_num = len(valid_index)
            cls_num += valid_num
            if not valid_num==0:
                unique_class[ii] = 1
            label_num += len(np.where(box_label[valid_index,ii]==1)[0])
            unique_class_len[ii+1] = valid_num + unique_class_len[ii]

        # origin box_box and box_score
        box_box_origin = box_box.copy()
        box_score_origin = box_score.copy()
        
        box_box[:, 0::2] = np.log(box_box[:, 0::2] / float(width) + 0.5)
        box_box[:, 1::2] = np.log(box_box[:, 1::2] / float(height) + 0.5)

        all_class_box_feature = np.zeros((cls_num, 1024))
        all_class_box_weight = np.ones((cls_num, 1))
        all_class_box_label = np.zeros((cls_num, 1))
        all_class_box_box = np.zeros((cls_num, 32))
        all_class_box_origin_box = np.zeros((cls_num, 4))
        all_class_box_origin_score = np.zeros((cls_num, 32))
        all_class_box_score = np.zeros((cls_num, 32))
        # logger = solver_log(os.path.join('/mnt/lustre/liushu1/qilu_ex/debug.log'))
        # logger.info(cls_num)
        # print()
        for ii in range(80):
            if unique_class[ii] == 0:
                continue
            start = int(unique_class_len[ii])
            end = int(unique_class_len[ii+1])
            valid_index = cls_list[ii]

            temp = box_score[valid_index, ii].argsort()[::-1]
            valid_sort = valid_index[temp]
            all_class_box_weight[start:end, 0] = box_weight[valid_sort, ii]
            all_class_box_label[start:end, 0] = box_label[valid_sort, ii]
            all_class_box_box[start:end, :] = np.tile(box_box[valid_sort, ii*4:(ii+1)*4], 8)
            all_class_box_origin_score[start:end, :] = np.tile(box_score_origin[valid_sort, ii:ii+1], 32)
            all_class_box_score[start:end, :] = self.search_table[0:end-start, 0:32]
            all_class_box_feature[start:end, :] = box_feature[valid_sort, :]
            all_class_box_origin_box[start:end, :] = box_box_origin[valid_sort, ii*4:(ii+1)*4]

        return all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, all_class_box_origin_box, gts_box, unique_class, unique_class_len, img_id
        

def unique_collate(batch):
    aa = [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0]))]
    # aa.extend(torch.IntTensor(batch[0][-1]))
    return aa

if __name__ == '__main__':
    base_path = '/data/luqi/dataset/pytorch_data/'
    img_list = 'train.txt'
    use_mode = 'unique'
    train = TrainDataset(base_path, img_list, weight=4, phase='train')
    num = len(train)
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    for i in range(num):
       all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, unique_class, unique_class_len, phase_np = train[i]
       # print(all_class_box_feature.shape[0])
       if i==20:
            break
       for ii in range(80):
            if unique_class[ii] == 0:
                continue
            start = int(unique_class_len[ii])
            end = int(unique_class_len[ii+1])
            # print(ii)
            # print(np.concatenate((all_class_box_origin_score[start:end, 0].reshape(-1, 1), all_class_box_label[start:end, 0].reshape(-1, 1), all_class_box_box[start:end, 0:4].reshape(-1, 4)), axis=1))
            # input()