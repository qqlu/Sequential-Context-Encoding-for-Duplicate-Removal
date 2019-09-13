from torch import optim
from torch.autograd import Variable
from torch import nn
import time
from .io import save_checkpoint, load_checkpoint
import os
import logging
import sys
from utils import solver_log
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
import torch

def train(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_label_variable, all_class_box_origin_score_variable, all_class_box_origin_box_variable, all_class_box_weight_tensor, gts_box_tensor, unique_class_cuda, unique_class_len_cuda, model, model_optimizer, pre_post_weight):
    model_optimizer.zero_grad()
    # input_length = box_feature_variable.size()[0]
    pre_stage_output, post_stage_output, post_stage_label, post_stage_weight, post_stage_box_origin_score_variable, post_stage_box_origin_box_tensor, post_unique_class, post_unique_class_len = model(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_origin_score_variable, all_class_box_origin_box_variable, gts_box_tensor, unique_class_cuda, unique_class_len_cuda)

    # calculate loss
    pre_criterion = nn.BCELoss(weight=all_class_box_weight_tensor)
    post_criterion = nn.BCELoss(weight=post_stage_weight)
    pre_loss = pre_criterion(pre_stage_output.view(-1, 1), all_class_box_label_variable)
    post_loss = post_criterion(post_stage_output, post_stage_label)
    loss = pre_loss + pre_post_weight * post_loss
    loss.backward()
    model_optimizer.step()

    # calculate presicion and recall
    pre_stage = pre_stage_output.data.cpu().numpy().reshape(-1, 1)
    pre_stage = (pre_stage > 0.5).astype(np.int8)
    pre_label = all_class_box_label_variable.data.cpu().numpy().astype(np.int8)
    post_stage = post_stage_output.data.cpu().numpy().reshape(-1, 1)
    post_stage = (post_stage > 0.5).astype(np.int8)
    post_label = post_stage_label.data.cpu().numpy().astype(np.int8)
    
    # print(box_label_np.shape)
    # print(output_record_np.shape)
    # input()
    pre_precision, pre_recall, _ , _ = score(pre_label, pre_stage, labels=[1])
    post_precision, post_recall, _ , _ = score(post_label, post_stage, labels=[1])
    precision = [float(pre_precision), float(post_precision)]
    recall = [float(pre_recall), float(post_recall)]
    loss = [pre_loss.data[0], post_loss.data[0]]
    torch.cuda.empty_cache()
    return loss, precision, recall


def solver(model, data_loader, n_epochs, output_dir, print_every=1, save_every=1, learning_rate=0.01, step=10, pre_post_weight=1, load_file=None, continue_epochs=None, continue_iters=None):
    # plot_losses = []
    if continue_epochs is None:
        continue_epochs = -1
        continue_iters = -1
        first_count = None
    else:
        first_count = continue_iters

    print_pre_loss_total = 0  # 
    print_post_loss_total = 0  #
    print_precision_total = [0, 0]
    print_recall_total = [0, 0]
    print_time_total = 0

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model_dir = output_dir[0]
    log_dir = output_dir[1]
    # log
    
    logger = solver_log(os.path.join(log_dir, 'train_'+ time.strftime('%Y%m%d_%H%M%S', time.localtime()) +'.log'))
    
    for epoch_index in range(n_epochs):
        # print len(data_loader)
        if (epoch_index < continue_epochs):
            continue
        if first_count is None:
            count = 0
        else:
            count = first_count
        for all_class_box_feature, all_class_box_box, all_class_box_score, all_class_box_label, all_class_box_weight, all_class_box_origin_score, all_class_box_origin_box, gts_box, unique_class, unique_class_len, img_id in data_loader:
            start = time.time()
            # print('begin')
            all_class_box_feature_variable =  Variable(all_class_box_feature).cuda()
            all_class_box_box_variable = Variable(all_class_box_box).cuda()
            all_class_box_score_variable = Variable(all_class_box_score).cuda()
            all_class_box_label_variable = Variable(all_class_box_label).cuda()
            all_class_box_origin_score_variable = Variable(all_class_box_origin_score).cuda()
            all_class_box_origin_box_variable = Variable(all_class_box_origin_box).cuda()
            all_class_box_weight_tensor = all_class_box_weight.cuda()
            gts_box_tensor = gts_box.cuda()
            # all_class_box_origin_box_variable = Variable(all_class_box_origin_box).cuda()
            # ranks = ranks.cuda()
            # logger.info(int(img_id.numpy()))
            image_id = int(img_id.numpy())
            unique_class_cuda = unique_class.cuda()
            unique_class_len_cuda = unique_class_len.cuda()

            loss, precision, recall = train(all_class_box_feature_variable, all_class_box_box_variable, all_class_box_score_variable, all_class_box_label_variable, all_class_box_origin_score_variable, all_class_box_origin_box_variable, all_class_box_weight_tensor, gts_box_tensor,
                        unique_class_cuda, unique_class_len_cuda, model, model_optimizer, pre_post_weight)
            count += 1
            print_pre_loss_total += loss[0]
            print_post_loss_total += loss[1]
            print_precision_total[0] += precision[0]
            print_precision_total[1] += precision[1]
            print_recall_total[0] += recall[0]
            print_recall_total[1] += recall[1]

            end = time.time()
            print_time_total += float(end-start)
            if count % print_every == 0:
                print_pre_loss_avg = print_pre_loss_total / print_every
                print_post_loss_avg = print_post_loss_total / print_every
                print_pre_pre_avg = print_precision_total[0] / print_every
                print_post_pre_avg = print_precision_total[1] / print_every
                print_pre_rec_avg = print_recall_total[0] / print_every
                print_post_rec_avg = print_recall_total[1] / print_every
                print_time_avg = print_time_total / print_every

                print_pre_loss_total = 0
                print_post_loss_total = 0
                print_precision_total = [0, 0]
                print_recall_total = [0, 0]
                print_time_total = 0
                # print('aa')
                logger.info('epoch:{}, iter:{}, lr:{}, avg_time:{:.3f}, pre_loss:{:.10f}, post_loss:{:.10f}, pre_pre:{:.3f}, post_pre:{:.3f}, pre_rec:{:.3f}, post_rec:{:.3f}'.format(epoch_index, count, learning_rate, print_time_avg, print_pre_loss_avg, print_post_loss_avg, print_pre_pre_avg, print_post_pre_avg, print_pre_rec_avg, print_post_rec_avg))
            # if count % save_every == 0:
                # save_checkpoint(model, epoch_index, count, model_dir)
        save_checkpoint(model, epoch_index, count, model_dir)

        print_loss_total = 0
        print_pos_acc_total = 0
        print_neg_acc_total = 0
        print_time_total = 0

        if epoch_index % step == 0 and epoch_index > 0:
            learning_rate = learning_rate * 0.1
            for param_group in model_optimizer.param_groups:
                param_group['lr'] = learning_rate
