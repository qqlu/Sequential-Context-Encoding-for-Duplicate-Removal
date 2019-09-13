import torch, cv2
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import os.path as osp
from utils import read_txt, visualize_from_txt_without_nms
from utils import assign_label, assign_label_with_nms
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
    def __init__(self, base_path, txt_name, use_mode, cls_list, overlap_thresh=0.5, score_thresh=0.01, phase='train'):
        super(TrainDataset, self).__init__(base_path, txt_name)
        # The use_mode is unique or msra
        self.use_mode = use_mode
        self.cls_list = cls_list
        self.cls_num = len(cls_list)
        self.overlap_thresh = overlap_thresh
        self.score_thresh = score_thresh
        self.phase = phase

    def __getitem__(self, idx):
        # logger1 = solver_log(os.path.join('/mnt/lustre/liushu1/', 'load_dataset.log'))
        info_pkl_path = os.path.join(self.info_path, self.image_index[idx] + '.pkl')
        img_path = os.path.join(self.img_path, self.image_index[idx] + '.jpg')
        img = cv2.imread(img_path)
        height, width = img.shape[0:2]
            
        box_feature, box_box, box_score = pkl.load(open(os.path.join(info_pkl_path), 'rb'), encoding='iso-8859-1')

        box_score_max = np.max(box_score[:,1:], axis=1)
        valid_index = np.where(box_score_max>=self.score_thresh)[0]
        box_feature = box_feature[valid_index, :]
        
        box_box = box_box[valid_index, :]
        box_box_origin = box_box.copy()
        box_box[:, 0::2] = box_box[:, 0::2] / width
        box_box[:, 1::2] = box_box[:, 1::2] / height
        box_score = box_score[valid_index, :]

        # d2_end = time.time()
        # d3_start = time.time()

        # gts_info = cvb.load(os.path.join(self.gt_path, self.image_index[idx] + '.pkl'))
        gts_info = pkl.load(open(os.path.join(self.gt_path, self.image_index[idx] + '.pkl'), 'rb'), encoding='iso-8859-1')
        gts_box = np.zeros((len(gts_info), 5))
        for index, gt in enumerate(gts_info):
            gts_box[index, :] = gt['bbox']
        # d3_end = time.time()

        # box_label= assign_label(box_box_origin, gts_box)
        box_label = assign_label(box_box_origin, gts_box)
        # box_label = assign_label(box_box_origin, box_score)

        num_box = box_feature.shape[0]
        ranks = np.empty_like(box_score)
        for ii in range(box_score.shape[1]):
            temp = box_score[:, ii].argsort()
            ranks[temp, ii] = np.arange(box_score.shape[0], 0, -1)
        rank_score = np.zeros((num_box, 81*32))
        for ii in range(ranks.shape[0]):
            for jj in range(ranks.shape[1]): 
                rank_score[ii, jj*32: (jj+1)*32] = self.search_table[int(ranks[ii, jj]), :]
        
        # logger1.info('d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(float(d1_end-d1_start), float(d2_end-d2_start), float(d3_end-d3_start)))
        return box_feature, rank_score, box_box, box_label
        # return box_feature, rank_score, box_box, box_label

def unique_collate(batch):
    #print len(batch)
    if len(batch[0]) == 3:
        return torch.LongFloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2])
    elif len(batch[0]) == 4:
        return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3])
    elif len(batch[0]) == 5:
        return torch.FloatTensor(batch[0][0]), torch.FloatTensor(batch[0][1]), torch.FloatTensor(batch[0][2]), torch.FloatTensor(batch[0][3]), torch.FloatTensor(batch[0][4])

if __name__ == '__main__':
    base_path = '/data/dataset/LSTM_data/'
    txt_name = 'train.txt'
    use_mode = 'unique'

    cls_list = ['_' for _ in range(81)]
    train = TrainDataset(base_path, txt_name, use_mode, cls_list)
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, num_workers=1, pin_memory=False)
    # for data_batch in train_loader:
        # print data_batch
        # raw_input()