from torch import optim
from torch.autograd import Variable
from torch import nn
import time
from .io import load_checkpoint
import os
import logging
import sys
from utils import solver_log
from .io import load_checkpoint
import torch
import cvbase as cvb
import numpy as np
import cv2
from pandas.core.frame import DataFrame

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

def highlight_greaterthan(s, threshold, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] >= threshold
#     print(type(is_max))
    # return "'color: %s' % color"
    return ['background-color: yellow' if is_max.any() else '' for v in is_max]


def test(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, unique_class_cuda, unique_class_len_cuda, model):

    output_record = model(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, unique_class_cuda, unique_class_len_cuda)
    # output_record = model(box_feature_variable, box_score_variable, box_box_variable, all_class_box_feature_variable, all_class_box_score_variable, all_class_box_box_variable, unique_class, unique_class_len)
    output_record_np = output_record.data.cpu().numpy().reshape(-1, 1).astype(np.float)

    return output_record_np


def test_solver(model, dataset, output_dir, thread_index, thread_num):
    # load checkpoint
    load_checkpoint(model, output_dir[0])
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Newlabel.pkl')
    # result_path = os.path.join(output_dir[1], 'result.json')
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    log_dir = output_dir[1]
    color_map = _get_voc_color_map()
    # count = 0
    logger = solver_log(os.path.join(log_dir, 'test_'+ time.strftime('%Y%m%d_%H%M%S', time.localtime()) +'.log'))
    # logger = solver_log(os.path.join(log_dir, 'test1.log'))
    results = []
    data_num = len(dataset)
    for count in range(data_num):
        if count % thread_num != thread_index:
            continue
        if count>=100:
            break
        data_np = dataset[count]
        # input
        # all_class_box_origin_score, all_class_box_origin_box, unique_class, unique_class_len, img_id
        # box_feature, rank_score, box_box = torch.FloatTensor(data_np[0]), torch.FloatTensor(data_np[1]), torch.FloatTensor(data_np[2])
        all_class_box_feature, all_class_box_box, all_class_box_score = torch.FloatTensor(data_np[0]), torch.FloatTensor(data_np[1]), torch.FloatTensor(data_np[2])
        all_class_box_label = data_np[3]
        all_class_box_weight = data_np[4]
        all_class_box_origin_score, all_class_box_origin_box = torch.FloatTensor(data_np[5]), data_np[6]
        unique_class, unique_class_len = torch.FloatTensor(data_np[7]), torch.FloatTensor(data_np[8])
        unique_class_np, unique_class_len_np = data_np[7], data_np[8]
        image_id = int(data_np[9])
        img_name = str(image_id).zfill(12)
        im_file = '/mnt/lustre/liushu1/qilu_ex/dataset/coco/fpn_bn_base/img/' + img_name + '.jpg'
        im = cv2.imread(im_file)
        bboxes = []
        start = time.time()

        # all_class_box_label_variable = Variable(all_class_box_label).cuda()
        all_class_box_score_variable = Variable(all_class_box_score).cuda()
        all_class_box_box_variable = Variable(all_class_box_box).cuda()
        all_class_box_feature_variable = Variable(all_class_box_feature).cuda()
        all_class_box_origin_score_variable = Variable(all_class_box_origin_score).cuda()

        unique_class_cuda = unique_class.cuda()
        unique_class_len_cuda = unique_class_len.cuda()

        output = test(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, unique_class_cuda, unique_class_len_cuda, model)

        box_score_origin = all_class_box_origin_score_variable.data.cpu().numpy().astype(np.float)[:,0:1].reshape(-1, 1)
        # final_score = box_score_origin
        # final_score = (box_score_origin + output) / 2
        final_score = box_score_origin * output
        for cls_index in range(80):
            if unique_class_np[cls_index] == 0:
                continue
            start_ = int(unique_class_len_np[cls_index])
            end_ = int(unique_class_len_np[cls_index+1])
            # info_info = np.concatenate((box_score_origin[start_:end_, 0:1], output[start_:end_, 0:1], final_score[start_:end_,0:1], all_class_box_origin_box[start_:end_, 0:4].astype(np.int), all_class_box_label[start_:end_, 0:1]), axis=1)
            # qwe = DataFrame(info_info, columns=['score_origin', 'network', 'final', 'x1', 'y1', 'x2', 'y2', 'label'])
            # print(qwe)
            # qwe.style.apply(highlight_greaterthan,threshold=0.5,column=['label'], axis=1)
            # qwe
            # print(qwe.to_string())
            # print(qwe.sort_values(by='final'))
            # print(qwe.sort_values(by='label'))
            # input()
            for index in range(start_, end_):
                x1, y1, x2, y2 = all_class_box_origin_box[index, 0:4]
                score = final_score[index, 0]
                # if(score<0.05):
                #     continue
                category_id = cls_index+1
                bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
        
        for bbox_single in bboxes:
            cls_indx = int(bbox_single['category_id'])
            x1, y1, w, h = bbox_single['bbox']
            score = bbox_single['score']
            if score<0.01:
                continue           
            cv2.rectangle(im, (int(x1), int(y1)), (int(x1+w-1), int(y1+h-1)), tuple(color_map[cls_indx, :]), 2)
        # count += 1
        save_path = os.path.join('/mnt/lustre/liushu1/qilu_ex/Post_vis/', img_name+'.jpg')
        # save_gt_path = os.path.join(args.output_dir, img_name+'.jpg')
        # cv2.imwrite('/data/luqi/000000156500_proposal.png', im_proposal)
        cv2.imwrite(save_path, im)
        end = time.time()
        print_time = float(end-start)
        # results.extend(bboxes)
        # if count==20:
            # break
        logger.info('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, count, image_id, print_time))
    return results
    # cvb.dump(results, result_path)

        
