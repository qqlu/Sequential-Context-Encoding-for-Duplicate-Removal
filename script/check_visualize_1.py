import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from evaluation import COCO
from evaluation import COCOeval
# from models.rnn_model_full_bire_class_split_add_origin_score_no_encoder import Encoder_Decoder
from dataset.multi_dataset_before_before_nms import TrainDataset, unique_collate
# from solver.solver_full_nms_weight import solver, load_checkpoint
import torch
import argparse
import os
import numpy as np
from multiprocessing import Process, Manager
import cvbase as cvb
import cv2
import time
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser(description='visualize')

    parser.add_argument('--base_path', 
                        default='/data/luqi/dataset/pytorch_data/',
                        help='the data path of RNN')

    parser.add_argument('--gt_path', 
                        default='/data/luqi/dataset/coco/annotations/instances_val2017.json',
                        help='the path of gt json')

    parser.add_argument('--img_list',
                        default='val.txt',
                        help='the img_list')

    parser.add_argument('--use_mode',
                        default='unique',
                        help='the method of score_box_fusion')

    parser.add_argument('--ann_type',
                        default='bbox',
                        help='the type of anns, det or segm')

    parser.add_argument('--thread_all', 
                        default=16,
                        type=int,
                        help='the hidden size of RNN')
    
    args = parser.parse_args()
    return args

def vis_detections(img, cls_name, cls_bbox):
    """Draw detected bounding boxes."""
    cls_len = len(cls_name)
    import math
    row_len = math.ceil(cls_len ** (0.5))
    col_len = math.ceil(cls_len / row_len)
    # fig, ax = plt.subplots(figsize=(12, 12))
    for ii in range(cls_len):
        im = img
        dets = cls_bbox[ii]
        name = cls_name[ii]    
        inds = dets.shape[0]

        im = im[:, :, (2, 1, 0)]
        fig = plt.subplot(row_len, col_len, ii+1)
        plt.imshow(im)
        fig.set_title(name)
        for i in range(inds):
            bbox = dets[i, :4]
            fig.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                              edgecolor='green', linewidth=1))
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    cls_list = ['_' for _ in range(81)]
    # datasets
    val = TrainDataset(args.base_path, args.img_list, args.use_mode, cls_list, phase='test')
    num = len(val)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    New2Old = cvb.load('/data/luqi/coco-master/PythonAPI/Newlabel.pkl')
    for i in range(num):
       all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, all_class_box_origin_box, unique_class, unique_class_len, image_id, phase_np = val[i]
       im_file = os.path.join(args.base_path, 'img/'+ str(image_id).zfill(12)+'.jpg')
       im = cv2.imread(im_file)
       # bboxes = []
       valid_num = 0
       all_num = unique_class_len[80]
       cls_num = 0
       cls_all_num = 0
       cls_info = []
       cls_name = []
       cls_bbox = []
       for cls_index in range(80):
            if unique_class[cls_index] == 0:
                continue
            # print(New2Old[str(cls_index+1)][0])
            cls_all_num += 1
            start = int(unique_class_len[cls_index])
            end = int(unique_class_len[cls_index+1])
            if(all_class_box_label[start, 0]==1):
                cls_num += 1
                valid_num += end - start
            bboxes = all_class_box_origin_box[start:end, 0:4]
            cls_bbox.append(bboxes)
            cls_name.append(New2Old[str(cls_index+1)][0])
            cls_info.append([New2Old[str(cls_index+1)][0], int(all_class_box_label[start, 0]), end - start, np.max(all_class_box_origin_score[start:end, 0]), np.mean(all_class_box_origin_score[start:end, 0])]) 
            # cvb.draw_bboxes(img, bboxes, win_name='aa')
            
            # print(np.concatenate((all_class_box_origin_score[start:end, 0].reshape(-1, 1), all_class_box_label[start:end, 0].reshape(-1, 1), all_class_box_origin_box[start:end, 0:4].reshape(-1, 4)), axis=1))
       print('image_id:{}, proposal:{}/{},{}, class:{}/{},{}'.format(image_id, valid_num, all_num, valid_num/all_num, cls_num, cls_all_num, cls_num/cls_all_num))
       print(DataFrame(cls_info, columns=['class','gt','num', 'max_score', 'mean_score']))
       vis_detections(im, cls_name, cls_bbox)
       
       input()
            # for index in range(start, end):
            #     if(all_class_box_label[index]==0):
            #         continue
            #     x1, y1, x2, y2 = all_class_box_origin_box[index, 0:4]
            #     score = all_class_box_origin_score[index, 0]
            #     category_id = New2Old[str(cls_index+1)][1]
            #     bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
            # count += 1
       # thread_result.extend(bboxes)
       # end_time = time.time()
       # print_time = float(end_time-start_time)
       # print('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, i, image_id, print_time))
    # result.extend(thread_result)
