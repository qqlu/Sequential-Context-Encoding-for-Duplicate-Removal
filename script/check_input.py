import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
import os
import cvbase as cvb
from evaluation import COCO
from evaluation import COCOeval
from utils import verify_nms
from utils import verify_nms_with_limit
from utils import verify_soft_nms
from utils import verify_nms_with_box_voting
import pickle as pkl
import numpy as np
from multiprocessing import Process, Manager
def parse_args():
    parser = argparse.ArgumentParser(description='Check Input')

    parser.add_argument('--base_path', 
                        default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/fpn_bn_base/',
    					# default='/data/luqi/dataset/coco_mask_rcnn/',
    					help='the data path of RNN')

    parser.add_argument('--gt_path', 
                        default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/annotations/instances_val2017.json',
                        help='the path of gt json')

    parser.add_argument('--img_list',
    					default='val.txt',
    					help='the img_list')

    parser.add_argument('--output_dir',
    					default='/mnt/lustre/liushu1/qilu_ex/check_9.json',
    					help='the save path of output_dir')

    parser.add_argument('--ann_type',
                        default='bbox',
                        help='the type of anns, det or segm')

    parser.add_argument('--thread_all', 
                        default=24,
                        type=int,
                        help='the hidden size of RNN')
    
    args = parser.parse_args()
    return args

def run(thread_index, thread_num, result, args):
    txt_path = os.path.join(args.base_path, args.img_list)
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Newlabel.pkl')
    proposal_base_path = os.path.join(args.base_path, 'info/')
    thread_result = []
    val_list = cvb.list_from_file(txt_path)
    val_num = len(val_list)
    for index, img_name in enumerate(val_list):
        if index % thread_num != thread_index:
            continue
        proposal_path = os.path.join(proposal_base_path, img_name + '.pkl')
        # box_feature, box_box, box_score = cvb.load(proposal_path)
        box_feature, box_box, box_score = pkl.load(open(os.path.join(proposal_path), 'rb'), encoding='iso-8859-1')
        box_feature = box_feature.astype(np.float)
        box_box = box_box.astype(np.float)
        box_score = box_score.astype(np.float)
        # bbox = verify_nms(box_box, box_score, img_name, New2Old, iou_thr=0.5, score_thr=0.1)
        # bbox = verify_nms_with_box_voting(box_box, box_score, img_name, New2Old, iou_thr=0.5, score_thr=0.01, bv_method='ID')
        bbox = verify_soft_nms(box_box, box_score, img_name, New2Old, iou_thr=0.5, score_thr=0.01)
        # bbox = verify_nms_with_limit(box_box, box_score, img_name, New2Old, iou_thr=0.5)
        # cls_num = 0
        # for ii in range(1, 81):
        #     valid_index = np.where(box_score[:,ii]>=0.01)[0]
        #     valid_num = len(valid_index)
        #     cls_num += valid_num

        thread_result.extend(bbox)
        print('img_index:{}, bbox_len:{}'.format(img_name, len(bbox)))
        # print('thread_index:{}, index:{}'.format(thread_index, index))
    result.extend(thread_result)

if __name__ == '__main__':
    with Manager() as manager:
        args = parse_args()
        # np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
        result_path = args.output_dir
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

