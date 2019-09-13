from torch import optim
from torch.autograd import Variable
from torch import nn
import time
import os
import logging
import sys
from utils import solver_log
import torch
import cvbase as cvb
import numpy as np
from pandas.core.frame import DataFrame

def test_solver(model, dataset, output_dir, thread_index, thread_num):
    # load checkpoint
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
        all_class_box_origin_score, all_class_box_origin_box = torch.FloatTensor(data_np[5]), torch.FloatTensor(data_np[6])
        gts_box = torch.FloatTensor(data_np[7])
        unique_class, unique_class_len = torch.FloatTensor(data_np[8]), torch.FloatTensor(data_np[9])
        pre_unique_class, pre_unique_class_len = data_np[8], data_np[9]
        image_id = int(data_np[10])
        # if data_np[1].shape[0]==0:
        #     results.extend(image_id)
        bboxes = []
        start = time.time()

        # all_class_box_label_variable = Variable(all_class_box_label).cuda()
        all_class_box_score_variable = Variable(all_class_box_score).cuda()
        all_class_box_box_variable = Variable(all_class_box_box).cuda()
        all_class_box_feature_variable = Variable(all_class_box_feature).cuda()
        all_class_box_origin_score_variable = Variable(all_class_box_origin_score).cuda()
        all_class_box_origin_box_variable = Variable(all_class_box_origin_box).cuda()
        gts_box_tensor = gts_box.cuda()
        unique_class_cuda = unique_class.cuda()
        unique_class_len_cuda = unique_class_len.cuda()

        pre_stage_output, post_stage_output, post_stage_label, post_stage_weight, post_stage_box_origin_score_variable, post_stage_box_origin_box_tensor, post_unique_class, post_unique_class_len = model(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, all_class_box_origin_box_variable, gts_box_tensor, unique_class_cuda, unique_class_len_cuda)
        # output_record = model(box_feature_variable, box_score_variable, box_box_variable, all_class_box_feature_variable, all_class_box_score_variable, all_class_box_box_variable, unique_class, unique_class_len)
        pre_output = pre_stage_output.data.cpu().numpy().reshape(-1, 1).astype(np.float)
        # output = test(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, all_class_box_origin_box_variable, gts_box_tensor, unique_class_cuda, unique_class_len_cuda, model)

        pre_score = all_class_box_origin_score_variable.data.cpu().numpy().astype(np.float)[:,0:1].reshape(-1, 1)
        pre_box = all_class_box_origin_box_variable.data.cpu().numpy()
        pre_label = all_class_box_label
        # post
        post_score = post_stage_box_origin_score_variable.data.cpu().numpy().astype(np.float)[:,0:1].reshape(-1,1)
        post_box = post_stage_box_origin_box_tensor.cpu().numpy()
        post_output = post_stage_output.data.cpu().numpy().reshape(-1, 1).astype(np.float)
        post_unique_class_np = post_unique_class.cpu().numpy()
        post_unique_class_len_np = post_unique_class_len.cpu().numpy()
        post_label = post_stage_label.data.cpu().numpy()
        
        torch.cuda.empty_cache()
        # final_score = box_score_origin
        # final_score = (box_score_origin + output) / 2
        pre_flag=False
        if pre_flag:
            final_score = pre_score * pre_output
            unique_class_np = pre_unique_class
            unique_class_len_np = pre_unique_class_len
            final_box = pre_box
            final_label = pre_label
        else:
            final_score = post_score * post_output
            unique_class_np = post_unique_class_np
            unique_class_len_np = post_unique_class_len_np
            final_box = post_box
            final_label = post_label

        # final_score = output
        for cls_index in range(80):
            if unique_class_np[cls_index] == 0:
                continue
            start_ = int(unique_class_len_np[cls_index])
            end_ = int(unique_class_len_np[cls_index+1])

            for index in range(start_, end_):
                x1, y1, x2, y2 = final_box[index, 0:4]
                score = final_score[index, 0]
                # if final_label[index, 0]==0:
                    # continue
                # if(score<0.01):
                    # continue
                category_id = New2Old[str(cls_index+1)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})

        end = time.time()
        print_time = float(end-start)
        results.extend(bboxes)
        logger.info('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, count, image_id, print_time))
    return results
    # cvb.dump(results, result_path)

        
