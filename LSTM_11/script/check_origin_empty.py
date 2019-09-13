import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
import argparse
import os
# import cvbase as cvb
import pickle as pkl
import numpy as np
import cvbase as cvb
import time
# sys.path.append('/data/luqi/LSTM/')
# sys.path.append('/mnt/lustre/liushu1/qilu_ex/LSTM/')
from utils.bbox_ops import bbox_overlaps
from multiprocessing import Process, Manager

def parse_args():
    parser = argparse.ArgumentParser(description='Check Input')

    parser.add_argument('--base_path', 
    					default='/mnt/lustre/liushu1/qilu_ex/dataset/test_dev/panet/',
    					help='the data path of RNN')

    parser.add_argument('--img_list',
    					default='test.txt',
    					help='the img_list')

    parser.add_argument('--thread_all', 
                        default=16,
                        type=int,
                        help='the hidden size of RNN')
    
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

def run(thread_index, thread_num, result, args):
    cls_list = ['_' for _ in range(81)]
    txt_path = os.path.join(args.base_path, args.img_list)
    all_index = cvb.list_from_file(txt_path)
    # datasets
    num = len(all_index)
    thread_result = []
    for i in range(num):
        if i % thread_num != thread_index:
            continue
        start_time = time.time()
        proposal_base_path = os.path.join(args.base_path, 'score/')
        img_index = all_index[i]
        image_id = int(img_index)
        proposal_path = os.path.join(proposal_base_path, img_index + '.pkl')
        if not os.path.exists(proposal_path):
            thread_result.append(img_index)
        # box_feature, box_box, box_score = pkl.load(open(os.path.join(proposal_path), 'rb'), encoding='iso-8859-1')
        # if np.size(box_feature)==0 or np.size(box_box)==0 or np.size(box_score)==0:
        #     thread_result.append(img_index)
        end_time = time.time()
        print_time = float(end_time-start_time)
        print('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, i, image_id, print_time))
    result.extend(thread_result)

if __name__ == '__main__':
    with Manager() as manager:
        args = parse_args()
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
        print(ori_result)


                    


