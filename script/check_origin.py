import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
import os
import cvbase as cvb
import pickle as pkl
import numpy as np
from evaluation import COCO
from evaluation import COCOeval
from gpu.cpu_nms_cython import nms

from utils.bbox_ops import bbox_overlaps
def parse_args():
    parser = argparse.ArgumentParser(description='Check Input')

    parser.add_argument('--base_path', 
    					default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/panet_dcn/',
    					help='the data path of RNN')

    parser.add_argument('--result_path',
                        default='/mnt/lustre/liushu1/qilu_ex/nips2018/data/result/panet_val_NMS.json',
                        help='the img_list')

    parser.add_argument('--gt_path', 
                        default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/annotations/instances_val2017.json',
                        help='the path of gt json')

    parser.add_argument('--img_list',
    					default='val.txt',
    					help='the img_list')

    parser.add_argument('--ann_type',
                        default='bbox',
                        help='the type of anns, det or segm')
    
    args = parser.parse_args()
    return args

def NMS_pure(key, box_box, box_score, New2Old, gts_box):
    bboxes = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((box_box[:, cls_index*4:(cls_index+1)*4], box_score[:, cls_index:cls_index+1]), axis=1)
        valid_index = np.where(cls_info[:, 4]>0.01)[0]
        cls_info = cls_info[valid_index, :]
        keep = nms(cls_info.astype(np.float32), 0.5)
        for final_index in list(keep):
            x1, y1, x2, y2, score = cls_info[final_index, :]
            category_id = New2Old[str(cls_index)][1]
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1)+1, int(y2-y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
    return bboxes

def NMS_score(key, box_box, box_score, New2Old, gts_box):
    unique_gts = np.unique(gts_box[:, 4]).astype(np.int16)
    bboxes = []
    for cls_index in unique_gts:
        if cls_index>100:
            continue
        cls_info = np.concatenate((box_box[:, cls_index*4:(cls_index+1)*4], box_score[:, cls_index:cls_index+1]), axis=1)
        valid_index = np.where(cls_info[:, 4]>0.01)[0]
        if len(valid_index)==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = nms(cls_info.astype(np.float32), 0.5)
        cls_info = cls_info[keep, :]

        gt_valid_index = np.where(gts_box[:, 4]==cls_index)[0]
        cls_gts_box = gts_box[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_info[:, :4], cls_gts_box)

        for col_index in range(ious.shape[1]):
            row_satis = np.where(ious[:, col_index] >= 0.5)[0]
            if len(row_satis)==0:
                continue
            arg_index = np.argmax(cls_info[row_satis, 4], axis=0)
            x1, y1, x2, y2, score = cls_info[row_satis[arg_index], :]
            category_id = New2Old[str(cls_index)][1]
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1)+1, int(y2-y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
    return bboxes

def NMS_iou(key, box_box, box_score, New2Old, gts_box):
    unique_gts = np.unique(gts_box[:, 4]).astype(np.int16)
    bboxes = []
    for cls_index in unique_gts:
        if cls_index>100:
            continue
        cls_info = np.concatenate((box_box[:, cls_index*4:(cls_index+1)*4], box_score[:, cls_index:cls_index+1]), axis=1)
        valid_index = np.where(cls_info[:, 4]>0.01)[0]
        if len(valid_index)==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = nms(cls_info.astype(np.float32), 0.5)
        
        cls_info = cls_info[keep, :]

        gt_valid_index = np.where(gts_box[:, 4]==cls_index)[0]
        cls_gts_box = gts_box[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_info[:, :4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > 0.5:
                arg_index = argmax_overlaps[index]
                x1, y1, x2, y2, score = cls_info[arg_index, :]
                if score<0.01:
                    continue
                category_id = New2Old[str(cls_index)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1)+1, int(y2-y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
    return bboxes

def no_removal(key, box_box, box_score, New2Old, gts_box):
    bboxes = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((box_box[:, cls_index*4:(cls_index+1)*4], box_score[:, cls_index:cls_index+1]), axis=1)
        box_valid_index = np.where(cls_info[:, 4]>0.01)[0]
        for final_index in box_valid_index:
            x1, y1, x2, y2, score = cls_info[final_index, :]
            category_id = New2Old[str(cls_index)][1]
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1)+1, int(y2-y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
    return bboxes

def no_removal_know_gt(key, box_box, box_score, New2Old, gts_box):
    unique_gts = np.unique(gts_box[:, 4]).astype(np.int16)
    bboxes = []
    for cls_index in unique_gts:
        if cls_index>100:
            continue
        cls_info = np.concatenate((box_box[:, cls_index*4:(cls_index+1)*4], box_score[:, cls_index:cls_index+1]), axis=1)
        box_valid_index = np.where(cls_info[:, 4]>0.01)[0]
        cls_info = cls_info[box_valid_index, :]
        
        gt_valid_index = np.where(gts_box[:, 4]==cls_index)[0]
        cls_gts_box = gts_box[gt_valid_index, 0:4]

        if cls_info.shape[0]==0:
            continue

        ious = bbox_overlaps(cls_info[:, :4], cls_gts_box)

        # max_overlaps = np.max(ious, axis=0)
        # argmax_overlaps= np.argmax(ious, axis=0)
        # for index in range(argmax_overlaps.shape[0]):
        #     if max_overlaps[index] > 0.5:
        #         arg_index = argmax_overlaps[index]
        #         x1, y1, x2, y2, score = cls_info[arg_index, :]
        #         if score<0.01:
        #             continue
        #         category_id = New2Old[str(cls_index)][1]
        #         bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1)+1, int(y2-y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})

        for col_index in range(ious.shape[1]):
            row_satis = np.where(ious[:, col_index] >= 0.5)[0]
            if len(row_satis)==0:
                continue
            arg_index = np.argmax(cls_info[row_satis, 4], axis=0)
            x1, y1, x2, y2, score = cls_info[row_satis[arg_index], :]
            category_id = New2Old[str(cls_index)][1]
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1)+1, int(y2-y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
    return bboxes

if __name__ == '__main__':
    args = parse_args()
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Newlabel.pkl')
    txt_path = os.path.join(args.base_path, args.img_list)
    all_index = cvb.list_from_file(txt_path)
    proposal_base_path = os.path.join(args.base_path, 'info/')
    results = []
    for count, img_index in enumerate(all_index):
        gts_info = pkl.load(open(os.path.join(args.base_path, 'gt/' + img_index + '.pkl'), 'rb'), encoding='iso-8859-1')
        gts_box = np.zeros((len(gts_info), 5))
        for index, gt in enumerate(gts_info):
            gts_box[index, :] = gt['bbox']

        # gts_box = None
        proposal_path = os.path.join(proposal_base_path, img_index + '.pkl')
        box_feature, box_box, box_score = pkl.load(open(os.path.join(proposal_path), 'rb'), encoding='iso-8859-1')
        
        box_feature = box_feature.astype(np.float)
        box_box = box_box.astype(np.float)
        box_score = box_score.astype(np.float)

        key = int(img_index)
        # bboxes = NMS_score(key, box_box, box_score, New2Old, gts_box)
        bboxes = NMS_pure(key, box_box, box_score, New2Old, gts_box)
        results.extend(bboxes)
        
        print('{}:{}'.format(count, key))
    cvb.dump(results, args.result_path)

    # # do evaluation
    cocoGt = COCO(args.gt_path)
    cocoDt = cocoGt.loadRes(args.result_path)

    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


                    


