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

def test(box_feature_variable, box_score_variable, box_box_variable, all_class_box_feature_variable, all_class_box_score_variable, all_class_box_box_variable, all_class_box_origin_score_variable, unique_class, unique_class_len, model):
    
    output_record, output_record_whole = model(box_feature_variable, box_score_variable, box_box_variable, all_class_box_feature_variable, all_class_box_score_variable, all_class_box_box_variable, all_class_box_origin_score_variable, unique_class, unique_class_len)
    # output_record = model(box_feature_variable, box_score_variable, box_box_variable, all_class_box_feature_variable, all_class_box_score_variable, all_class_box_box_variable, unique_class, unique_class_len)
    output_record_np = output_record.data.cpu().numpy().reshape(-1, 1).astype(np.float)

    return output_record_np


def test_solver(model, dataset, output_dir, thread_index, thread_num):
    # load checkpoint
    load_checkpoint(model, output_dir[0])
    New2Old = cvb.load('/mnt/lustre/liushu1/mask_rcnn/coco-master/PythonAPI/Newlabel.pkl')
    # result_path = os.path.join(output_dir[1], 'result.json')
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
        box_feature, rank_score, box_box = torch.FloatTensor(data_np[0]), torch.FloatTensor(data_np[1]), torch.FloatTensor(data_np[2])
        all_class_box_feature, all_class_box_box, all_class_box_score = torch.FloatTensor(data_np[3]), torch.FloatTensor(data_np[4]), torch.FloatTensor(data_np[5])
        all_class_box_label = data_np[6]
        all_class_box_class, all_class_box_origin_score, all_class_box_origin_box = data_np[7], torch.FloatTensor(data_np[8]), data_np[9]
        unique_class, unique_class_len = torch.FloatTensor(data_np[10]), torch.FloatTensor(data_np[11])
        image_id = int(data_np[12])

        bboxes = []
        start = time.time()
        box_feature_variable =  Variable(box_feature).cuda()
        box_score_variable = Variable(rank_score).cuda()
        # box_label_variable = Variable(box_label).cuda()
        box_box_variable = Variable(box_box).cuda()

        # all_class_box_label_variable = Variable(all_class_box_label).cuda()
        all_class_box_score_variable = Variable(all_class_box_score).cuda()
        all_class_box_box_variable = Variable(all_class_box_box).cuda()
        all_class_box_feature_variable = Variable(all_class_box_feature).cuda()
        all_class_box_origin_score_variable = Variable(all_class_box_origin_score).cuda()

        unique_class_cuda = unique_class.cuda()
        unique_class_len_cuda = unique_class_len.cuda()

        output = test(box_feature_variable, box_score_variable, box_box_variable, all_class_box_feature_variable, all_class_box_score_variable, all_class_box_box_variable, all_class_box_origin_score_variable, unique_class_cuda, unique_class_len_cuda, model)

        box_score_origin = all_class_box_origin_score_variable.data.cpu().numpy().astype(np.float)[:,0:1].reshape(-1, 1)
        # final_score = box_score_origin
        final_score = box_score_origin * output

        for index in range(final_score.shape[0]):
            cls_index = int(all_class_box_class[index, 0])
            x1, y1, x2, y2 = all_class_box_origin_box[index, 0:4]
            score = final_score[index, 0]
            category_id = New2Old[str(cls_index+1)][1]
            bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
        # count += 1
        end = time.time()
        print_time = float(end-start)
        results.extend(bboxes)
        logger.info('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, count, image_id,print_time))
    return results
    # cvb.dump(results, result_path)

        
