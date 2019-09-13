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

def test(box_feature_variable, box_score_variable, box_box_variable, model):

    input_length = box_feature_variable.size(0)
    # output_record_class, output_record_final = model(box_feature_variable, box_score_variable, box_box_variable, input_length)
    output_record_class, output_record_final = model(box_feature_variable, box_score_variable, box_box_variable, input_length)
    output_record_final_final = output_record_class * output_record_final 
    # output_record = model(box_feature_variable, box_score_variable, box_box_variable, input_length)
    output_record_np = output_record_final_final.data.cpu().numpy().reshape(-1, 1).astype(np.float)

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
            # count += 1
            continue
        data_np = dataset[count]
        # print(type(data_np))
        # print(len(data_np))
        # input()
        box_feature, rank_score, box_box, box_label, box_score_origin, box_box_origin, image_id, box_keep_np = torch.FloatTensor(data_np[0]), torch.FloatTensor(data_np[1]), torch.FloatTensor(data_np[2]), torch.FloatTensor(data_np[3]), torch.FloatTensor(data_np[4]), torch.FloatTensor(data_np[5]), torch.IntTensor([data_np[6]]), torch.FloatTensor(data_np[7])
        # print(image_id)
        # input()
        image_id = int(image_id.numpy())
        bboxes = []
        start = time.time()
        box_feature_variable =  Variable(box_feature).cuda()
        box_score_variable = Variable(rank_score).cuda()
        # box_label_variable = Variable(box_label).cuda()
        box_box_variable = Variable(box_box).cuda()

        output = test(box_feature_variable, box_score_variable, box_box_variable, model)
        # keep = list(np.where(output==1)[0])
        box_score_origin = box_score_origin.cpu().numpy().astype(np.float)
        box_keep_np = box_keep_np.cpu().numpy().astype(np.int)
        # final_score = box_score_origin * output
        final_score = box_score_origin * output

        # for index in keep:
        for index in range(final_score.shape[0]):
            # cls_index = np.argmax(box_score_origin[index, :])
            # if output[index, 0]==0:
                # continue
            cls_all_index = np.where(box_keep_np[index, :]==1)[0]
            for cls_index in cls_all_index:
                # cls_index = np.argsort(final_score[index, :])[::-1][0]
                x1, y1, x2, y2 = box_box_origin[index, cls_index*4:cls_index*4+4]
                # if abs(box_score_origin[index, cls_index]-output[index, 0]) >= 0.8:
                    # continue
                score = final_score[index, cls_index]
                # score = 1
                # score = box_score_origin[index, cls_index]
                category_id = New2Old[str(cls_index+1)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
        # count += 1
        end = time.time()
        print_time = float(end-start)
        results.extend(bboxes)
        logger.info('thread_index:{}, index:{}, image_id:{}, cost:{}'.format(thread_index, count, image_id,print_time))
    return results
    # cvb.dump(results, result_path)

        
