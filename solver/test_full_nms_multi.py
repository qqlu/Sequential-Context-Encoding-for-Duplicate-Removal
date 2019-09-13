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
from pandas.core.frame import DataFrame

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
    torch.cuda.empty_cache()
    return output_record_np


def test_solver(model, dataset, output_dir, thread_index, thread_num):
    # load checkpoint
    load_checkpoint(model, output_dir[0])
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Newlabel.pkl')
    # result_path = os.path.join(output_dir[1], 'result.json')
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    log_dir = output_dir[1]
    # count = 0
    logger = solver_log(os.path.join(log_dir, 'test_'+ time.strftime('%Y%m%d_%H%M%S', time.localtime()) +'.log'))
    # logger = solver_log(os.path.join(log_dir, 'test1.log'))
    results = []
    data_num = len(dataset)
    for count in range(data_num):
        if count % thread_num != thread_index:
            continue
        data_np = dataset[count]
        # input
        all_class_box_feature, all_class_box_box, all_class_box_score = torch.FloatTensor(data_np[0]), torch.FloatTensor(data_np[1]), torch.FloatTensor(data_np[2])
        all_class_box_label = data_np[3]
        all_class_box_weight = data_np[4]
        if all_class_box_weight.shape[0]==0:
            continue
        all_class_box_origin_score, all_class_box_origin_box = torch.FloatTensor(data_np[5]), data_np[6]
        unique_class, unique_class_len = torch.FloatTensor(data_np[7]), torch.FloatTensor(data_np[8])
        unique_class_np, unique_class_len_np = data_np[7], data_np[8]
        image_id = int(data_np[9])
        # if data_np[1].shape[0]==0:
        #     results.extend(image_id)
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
        # final_score = output
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
                # if(score<0.01):
                    # continue
                category_id = New2Old[str(cls_index+1)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})

        end = time.time()
        print_time = float(end-start)
        results.extend(bboxes)
        # if count==20:
            # break
        logger.info('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, count, image_id, print_time))
    return results
    # cvb.dump(results, result_path)

        
