import torch, cv2
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import os.path as osp
sys.path.append('/data/luqi/LSTM/')
# sys.path.append('/mnt/lustre/liushu1/qilu_ex/LSTM/')
from utils.bbox_ops import bbox_overlaps, stage2_nms_proposals
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
        self.pos_iou_thr = 0.5

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
            # if gts_box[index, 4]>=100:
                # gts_box[index, 4] -= 100

        unique_gts = np.unique(gts_box[:, 4]).astype(np.int16)

        keep_nms_np = stage2_nms_proposals(box_box, box_score)
        box_score = box_score[:, 1:]
        # box_feature = proposals_feature
        box_box = box_box[:, 4:]

        unique_class = np.zeros(80).astype(np.float)
        unique_class_len = np.zeros(81).astype(np.float)    # cumulative sum
        cls_num = 0
        cls_list = []
        for ii in range(80):
            valid_index = np.where(keep_nms_np[:, ii]==1)[0]
            cls_list.append(valid_index)
            valid_num = len(valid_index)
            cls_num += valid_num
            if not valid_num==0:
                unique_class[ii] = 1
            unique_class_len[ii+1] = valid_num + unique_class_len[ii]

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

        for ii in range(80):
            if unique_class[ii] == 0:
                continue
            start = int(unique_class_len[ii])
            end = int(unique_class_len[ii+1])
            
            valid_index = cls_list[ii]
            temp = box_score[valid_index, ii].argsort()[::-1]
            valid_sort = valid_index[temp]
            
            all_class_box_box[start:end, :] = np.tile(box_box[valid_sort, ii*4:(ii+1)*4], 8)
            all_class_box_origin_score[start:end, :] = np.tile(box_score_origin[valid_sort, ii:ii+1], 32)
            all_class_box_score[start:end, :] = self.search_table[0:end-start, 0:32]
            all_class_box_feature[start:end, :] = box_feature[valid_sort, :]
            all_class_box_origin_box[start:end, :] = box_box_origin[valid_sort, ii*4:(ii+1)*4]
            # assign label
            cls_index = ii + 1
            if cls_index in unique_gts:
                gt_valid_index = np.where(gts_box[:, 4]==cls_index)[0]
                cls_gts_box = gts_box[gt_valid_index, 0:4]
                cls_nms_info = all_class_box_origin_box[start:end, :]
                cls_nms_score = all_class_box_origin_score[start:end, 0:1]
                ious = bbox_overlaps(cls_nms_info, cls_gts_box)

                for col_index in range(ious.shape[1]):
                    row_satis = np.where(ious[:, col_index]>=self.pos_iou_thr)[0]
                    if len(row_satis)==0:
                        continue
                    argmax_satis = np.argmax(cls_nms_score[row_satis, 0], axis=0)
                    all_class_box_label[start+row_satis[argmax_satis], 0] = 1
                    all_class_box_weight[start+row_satis[argmax_satis], 0] = self.weight
                
                # max_overlaps = np.max(ious, axis=0)
                # argmax_overlaps= np.argmax(ious, axis=0)
        
                # for index in range(argmax_overlaps.shape[0]):
                #     if max_overlaps[index] > self.pos_iou_thr:
                #         arg_index = argmax_overlaps[index]
                #         all_class_box_label[start+arg_index, 0] = 1
                #         all_class_box_weight[start+arg_index, 0] = self.weight
        
        phase_np = np.zeros(1)
        if self.phase == 'train':
            phase_np[0] = 1
        if self.phase == 'bug':
            phase_np[0] = 2
        # class_split_score_feature or box_feature
        
        if self.phase == 'train':
            return all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, unique_class, unique_class_len, phase_np
        elif self.phase == 'test':
            return all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, all_class_box_origin_box, unique_class, unique_class_len, img_id, phase_np    

def unique_collate(batch):
    #print len(batch)
    if batch[0][-1][0]==1 or batch[0][-1][0]==2:
        # print(len(batch[0]))
        aa = [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0])-1)]
        return aa
    else:
        aa = [torch.FloatTensor(batch[0][i]) for i in range(len(batch[0])-2)]
        aa.extend(torch.IntTensor(batch[0][-2]))
        return aa

if __name__ == '__main__':
    base_path = '/data/luqi/dataset/pytorch_data/'
    img_list = 'val.txt'
    use_mode = 'unique'
    cls_list = ['_' for _ in range(81)]
    train = TrainDataset(base_path, img_list, use_mode, cls_list, weight=2, phase='test')
    num = len(train)
    New2Old = cvb.load('/data/luqi/coco-master/PythonAPI/Newlabel.pkl')
    np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
    results = []
    for i in range(num):
       all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, all_class_box_origin_box, unique_class, unique_class_len, img_id, phase_np = train[i]
       # if i==20:
       #      break
       image_id = int(img_id)
       bboxes = []
       for ii in range(80):
            if unique_class[ii] == 0:
                continue
            start = int(unique_class_len[ii])
            end = int(unique_class_len[ii+1])
            for index in range(start, end):
                if all_class_box_label[index, 0]==0:
                    continue
                x1, y1, x2, y2 = all_class_box_origin_box[index, 0:4]
                score = all_class_box_origin_score[index, 0]
                category_id = New2Old[str(ii+1)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
       results.extend(bboxes)
       print('{}:{}'.format(i, image_id))
    cvb.dump(results, '/data/luqi/check_2.json')
        # count += 1
        # end = time.time()
        # print_time = float(end-start)
        
        # print(ii)
        # print(np.concatenate((all_class_box_origin_score[start:end, 0].reshape(-1, 1), all_class_box_label[start:end, 0].reshape(-1, 1), all_class_box_origin_box[start:end, 0:4].reshape(-1, 4)), axis=1))
        # input()