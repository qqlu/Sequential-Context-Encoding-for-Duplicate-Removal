import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
import os
import cvbase as cvb
from utils import stage2_filter, stage2_assign
import pickle as pkl
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Stage 2')

    parser.add_argument('--base_path', 
    					default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/pytorch_data/',
    					help='the data path of RNN')

    parser.add_argument('--img_list',
    					default='val.txt',
    					help='the img_list')

    parser.add_argument('--output_dir',
    					default='/mnt/lustre/liushu1/qilu_ex/stage_4.json',
    					help='the save path of output_dir')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    txt_path = os.path.join(args.base_path, args.img_list)
    result_path = args.output_dir
    proposal_base_path = os.path.join(args.base_path, 'info_1/')
    gts_base_path = os.path.join(args.base_path, 'gt/')
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Newlabel.pkl')

    val_list = cvb.list_from_file(txt_path)
    val_num = len(val_list)
    result = []
    for val_index, img_name in enumerate(val_list):
        # proposal
        proposal_path = os.path.join(proposal_base_path, img_name + '.pkl')
        box_feature, box_box, box_score = pkl.load(open(os.path.join(proposal_path), 'rb'), encoding='iso-8859-1')
        # gt
        gts_info = pkl.load(open(os.path.join(gts_base_path, img_name + '.pkl'), 'rb'), encoding='iso-8859-1')
        gts_box = np.zeros((len(gts_info), 5))
        for index, gt in enumerate(gts_info):
            gts_box[index, :] = gt['bbox']

        box_box = box_box.astype(np.float)
        box_score = box_score.astype(np.float)
        box_feature = box_feature.astype(np.float)
        
        proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label = stage2_assign(box_feature, box_box, box_score, gts_box)
        proposals_score_nms = proposals_score_nms[:, 1:]
        proposals_box_nms = proposals_box_nms[:, 4:]
        valid_index = list(np.where(proposals_label==1)[0])

        bboxes = []
        image_id = int(img_name)
        for ii in valid_index:
            cls_index = np.argmax(proposals_score_nms[ii, :])
            score = proposals_score_nms[ii, cls_index]
            x1, y1, x2, y2 = proposals_box_nms[ii, cls_index*4:cls_index*4+4]
            category_id = New2Old[str(cls_index+1)][1]
            bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
        # nms_filter_count, nms_count = Calculate_ratio(box_box, box_score, gts_box, img_name, New2Old, iou_thr=0.3)
        
        result.extend(bboxes)
        print('{}/{}'.format(val_index, val_num))
        # input()
    cvb.dump(result, result_path)

