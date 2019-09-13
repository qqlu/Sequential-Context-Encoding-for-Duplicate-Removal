import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from evaluation import COCO
from evaluation import COCOeval
import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Do evaluation')

    parser.add_argument('--gt_path', 
    					default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/annotations/instances_val2017.json',
    					help='the path of gt json')
    parser.add_argument('--test_path',
                        default='/mnt/lustre/liushu1/qilu_ex/nips2018/data/result/fpn_test_NMS.json',
                        help='the path of test json')
    parser.add_argument('--ann_type',
                        default='bbox',
                        help='the type of anns, det or segm')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Running evaluation for {}'.format(args.ann_type))
    # initialization
    cocoGt = COCO(args.gt_path)
    cocoDt = cocoGt.loadRes(args.test_path)

    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()