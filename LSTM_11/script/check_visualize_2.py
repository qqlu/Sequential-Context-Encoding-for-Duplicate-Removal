import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
import os
import cvbase as cvb
from evaluation import COCO
from evaluation import COCOeval
from utils import verify_nms
# from utils import verify_nms_with_limit
from utils import verify_soft_nms
import pickle as pkl
import numpy as np
import cv2
def parse_args():
    parser = argparse.ArgumentParser(description='Check Input')

    parser.add_argument('--base_path', 
                        default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/fpn_bn_base/',
                        help='the data path of RNN')

    parser.add_argument('--gt_path', 
                        default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/annotations/instances_val2017.json',
                        help='the path of gt json')

    parser.add_argument('--img_list',
              default='val.txt',
              help='the img_list')

    parser.add_argument('--output_dir',
              default='/mnt/lustre/liushu1/qilu_ex/',
              help='the save path of output_dir')
    
    args = parser.parse_args()
    return args

def _get_voc_color_map(n=256):
    color_map = np.zeros((n, 3))
    for i in range(n):
        r = b = g = 0
        cid = i
        for j in range(0, 8):
            r = np.bitwise_or(r, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7-j))
            g = np.bitwise_or(g, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7-j))
            b = np.bitwise_or(b, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7-j))
            cid = np.right_shift(cid, 3)

        color_map[i][0] = b
        color_map[i][1] = g
        color_map[i][2] = r
    return color_map

if __name__ == '__main__':
    args = parse_args()
    txt_path = os.path.join(args.base_path, args.img_list)
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Oldlabel.pkl')
    proposal_base_path = os.path.join(args.base_path, 'info/')
    color_map = _get_voc_color_map()
    # color_map = color_map[np.random.shuffle(np.arange(256)), :].reshape(-1, 3)
    # print(color_map.shape)
    val_list = cvb.list_from_file(txt_path)
    # val_list = ['000000156500']
    val_num = len(val_list)
    for index, img_name in enumerate(val_list):
        if index>=100:
            break
        proposal_path = os.path.join(proposal_base_path, img_name + '.pkl')
        # box_feature, box_box, box_score = cvb.load(proposal_path)
        box_feature, box_box, box_score = pkl.load(open(os.path.join(proposal_path), 'rb'), encoding='iso-8859-1')
        box_feature = box_feature.astype(np.float)
        box_box = box_box.astype(np.float)
        box_score = box_score.astype(np.float)
        
        im_file = os.path.join(args.base_path, 'img/' + img_name + '.jpg')
        im = cv2.imread(im_file)
        # im_proposal = im.copy()
        im_nms = im.copy()
        im_gt = im.copy()

        bbox = verify_nms(box_box, box_score, img_name, New2Old, iou_thr=0.5, score_thr=0.01)
        for bbox_single in bbox:
             cls_indx = int(bbox_single['category_id'])
             x1, y1, w, h = bbox_single['bbox']
             score = bbox_single['score']           
             cv2.rectangle(im_nms, (int(x1), int(y1)), (int(x1+w-1), int(y1+h-1)), tuple(color_map[cls_indx, :]), 2)
             # cv2.putText(im_nms, "{}:{:.3f}".format(New2Old[str(cls_indx)][0], score), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        # (255, 255, 255), 1)
        gts_info = pkl.load(open(os.path.join(args.base_path, 'gt/' + img_name + '.pkl'), 'rb'), encoding='iso-8859-1')
        
        for index, gt in enumerate(gts_info):
            cls_indx = int(gt['bbox'][4])
            x1, y1, x2,y2 = gt['bbox'][0:4]
            cv2.rectangle(im_gt, (int(x1), int(y1)), (int(x2), int(y2)), tuple(color_map[cls_indx, :]), 2)
        # print(gts_box)
        save_nms_path = os.path.join(args.output_dir+'NMS_vis', img_name+'.jpg')
        save_gt_path = os.path.join(args.output_dir+'gt_vis', img_name+'.jpg')
        # save_gt_path = os.path.join(args.output_dir, img_name+'.jpg')
        # cv2.imwrite('/data/luqi/000000156500_proposal.png', im_proposal)
        cv2.imwrite(save_nms_path, im_nms)
        cv2.imwrite(save_gt_path, im_gt)
        # cv2.imwrite('/data/luqi/000000156500_gt.png', im_gt)
        # cv2.namedWindow('im_proposal', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('im_nms', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('im_gt', cv2.WINDOW_NORMAL)
        # cv2.imshow('im_proposal', im_proposal)
        # cv2.imshow('im_nms', im_nms)
        # cv2.imshow('im_gt', im_gt)
        # cv2.waitKey(0)

        # bbox = verify_nms(box_box, box_score, img_name, New2Old, iou_thr=0.5, score_thr=0.05)
        
