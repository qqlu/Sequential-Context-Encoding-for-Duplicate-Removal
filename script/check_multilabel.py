import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from evaluation import COCO
from evaluation import COCOeval
# from models.rnn_model_full_bire_class_split_add_origin_score_no_encoder import Encoder_Decoder
# from dataset.multi_dataset_before_before_no_gt_nms import TrainDataset, unique_collate
from dataset.multi_dataset_after_from_before_nms_msra import TrainDataset, unique_collate
# from solver.solver_full_nms_weight import solver, load_checkpoint
import torch
import argparse
import os
import numpy as np
from multiprocessing import Process, Manager
import cvbase as cvb
import time

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
                        default=8,
                        type=int,
                        help='the hidden size of RNN')
    
    args = parser.parse_args()
    return args

def run(thread_index, thread_num, result, args):
    cls_list = ['_' for _ in range(81)]
    # datasets
    val = TrainDataset(args.base_path, args.img_list, args.use_mode, cls_list, phase='test')
    num = len(val)
    New2Old = cvb.load('/data/luqi/coco-master/PythonAPI/Newlabel.pkl')
    multilabel = cvb.load('/data/luqi/dataset/pytorch_data/multilabel.pkl')
    
    thread_result = []
    for i in range(num):
       if i % thread_num != thread_index:
            continue
       start_time = time.time()
       all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, all_class_box_origin_box, unique_class, unique_class_len, image_id, phase_np = val[i]
       bboxes = []
       for cls_index in range(80):
            if unique_class[cls_index] == 0:
                continue
            img_name = str(int(image_id)).zfill(12)
            class_score = float(multilabel[img_name][cls_index])
            start = int(unique_class_len[cls_index])
            end = int(unique_class_len[cls_index+1])
            for index in range(start, end):
                # if(all_class_box_label[index]==0):
                    # continue
                x1, y1, x2, y2 = all_class_box_origin_box[index, 0:4]
                # score = all_class_box_origin_score[index, 0] * class_score
                score = all_class_box_origin_score[index, 0]
                category_id = New2Old[str(cls_index+1)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
            # count += 1
       thread_result.extend(bboxes)
       end_time = time.time()
       print_time = float(end_time-start_time)
       print('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, i, image_id, print_time))
    result.extend(thread_result)

if __name__ == '__main__':
    with Manager() as manager:
        args = parse_args()
        # np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
        result_path = '/data/luqi/check_after_from_before_multilabel.json'
        result = manager.list()
            
        p_list = []
        for i in range(args.thread_all):
            p = Process(target=run, args=(i, args.thread_all, result, args))
            p.start()
            p_list.append(p)

        for res in p_list:
            res.join()
        # print(result)
        ori_result = list(result)
        cvb.dump(ori_result, result_path)
        # do evaluation
        cocoGt = COCO(args.gt_path)
        cocoDt = cocoGt.loadRes(result_path)

        cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()