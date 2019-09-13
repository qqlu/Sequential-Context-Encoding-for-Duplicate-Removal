import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
import os
import cvbase as cvb
import pickle as pkl
import numpy as np
# sys.path.append('/data/luqi/LSTM/')
# sys.path.append('/mnt/lustre/liushu1/qilu_ex/LSTM/')
from utils.bbox_ops import bbox_overlaps
def parse_args():
    parser = argparse.ArgumentParser(description='Check Input')

    parser.add_argument('--base_path', 
    					default='/data/luqi/dataset/pytorch_data/',
    					help='the data path of RNN')

    parser.add_argument('--gt_path', 
                        default='/data/luqi/dataset/coco/annotations/instances_val2017.json',
                        help='the path of gt json')

    parser.add_argument('--img_list',
    					default='val.txt',
    					help='the img_list')

    parser.add_argument('--output_dir',
    					default='/data/luqi/RNN_NMS/before_mlp_source_4/result/result.json',
    					help='the save path of output_dir')

    
    args = parser.parse_args()
    return args

def make_json_dict(anns):
    anns_dict = {}
    for ann in anns:
        image_id = ann['image_id']
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)
    return anns_dict

def trans_np(ann_dict, Old2New):
    num = len(ann_dict)
    # x1, y1, x2, y2, score, label
    ann_np = np.zeros((num, 6), dtype=np.float)
    for index, ann in enumerate(ann_dict):
        x1, y1, width, height = ann['bbox']
        x2 = x1 + width - 1
        y2 = y1 + height - 1
        score = ann['score']
        cls_index = int(Old2New[str(ann['category_id'])][1])
        ann_np[index, :] = x1, y1, x2, y2, score, cls_index
    return ann_np

if __name__ == '__main__':
    args = parse_args()
    New2Old = cvb.load('/data/luqi/coco-master/PythonAPI/Newlabel.pkl')
    Old2New = cvb.load('/data/luqi/coco-master/PythonAPI/Oldlabel.pkl')
    all_anns = cvb.load(args.output_dir)
    anns_dict = make_json_dict(all_anns)
    txt_path = os.path.join(args.base_path, args.img_list)
    all_index = cvb.list_from_file(txt_path)
    results = []
    for count, img_index in enumerate(all_index):
        gts_info = pkl.load(open(os.path.join(args.base_path, 'gt/' + img_index + '.pkl'), 'rb'))
        gts_box = np.zeros((len(gts_info), 5))
        for index, gt in enumerate(gts_info):
            gts_box[index, :] = gt['bbox']
            # if gts_box[index, 4]>=100:
                # gts_box[index, 4] -= 100
        unique_gts = np.unique(gts_box[:, 4]).astype(np.int16)

        key = int(img_index)
        ann_dict = anns_dict[key]
        ann_matrix = trans_np(ann_dict, Old2New)
        bboxes = []
        for cls_index in unique_gts:
            gt_valid_index = np.where(gts_box[:, 4]==cls_index)[0]
            cls_gts_box = gts_box[gt_valid_index, 0:4]
            cls_where = np.where(ann_matrix[:, 5]==cls_index)[0]
            cls_info = ann_matrix[cls_where, :5]

            if cls_info.shape[0]==0:
                continue
            for arg_index in range(cls_info.shape[0]):
                x1, y1, x2, y2, score = cls_info[arg_index, :]
                category_id = New2Old[str(cls_index)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
            # ious = bbox_overlaps(cls_info[:, :4], cls_gts_box)

            # max_overlaps = np.max(ious, axis=0)
            # argmax_overlaps= np.argmax(ious, axis=0)
    
            # for index in range(argmax_overlaps.shape[0]):
            #     if max_overlaps[index] > 0.5:
            #         arg_index = argmax_overlaps[index]
            #         x1, y1, x2, y2, score = cls_info[arg_index, :]
            #         category_id = New2Old[str(cls_index)][1]
            #         bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(key)})
        results.extend(bboxes)
        print('{}:{}'.format(count, key))
    cvb.dump(results, '/data/luqi/check_7.json')


                    


