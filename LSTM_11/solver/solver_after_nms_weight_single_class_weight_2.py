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

def train(box_feature_variable, box_score_variable, box_box_variable, box_label_variable, box_class_label_variable, model, model_optimizer, criterion_final=None, criterion_class=None, pos_neg_weight=100):
    model_optimizer.zero_grad()

    input_length = box_feature_variable.size()[0]
    output_record_class, output_record_final = model(box_feature_variable, box_score_variable, box_box_variable, input_length)

    loss_class = criterion_class(output_record_class.view(-1, 1), box_class_label_variable)
    loss_final = criterion_final(output_record_final.view(-1, 1), box_label_variable)
    # loss_final = criterion_final()
    loss = loss_class + 1 * loss_final
    loss.backward()
    model_optimizer.step()

    # calculate presicion and recall
    output_record_np = output_record_final.data.cpu().numpy().reshape(-1, 1)
    output_record_np_class = output_record_class.data.cpu().numpy().reshape(-1, 1)
    # output_record_np = output_record_np_class * output_record_np

    output_record_np = (output_record_np > 0.5).astype(np.int8)

    box_label_np = box_label_variable.data.cpu().numpy().astype(np.int8)

    precision, recall, _ , _ = score(output_record_np, box_label_np, labels=[1])
    return loss.data[0], float(precision), float(recall)


def solver(model, data_loader, n_epochs, output_dir, print_every=1, save_every=1, learning_rate=0.01, step=10, pos_neg_weight=100, load_file=None, continue_epochs=None, continue_iters=None):
    # plot_losses = []
    if continue_epochs is None:
        continue_epochs = -1
        continue_iters = -1
        first_count = None
    else:
        first_count = continue_iters
    print_losses = []
    print_loss_total = 0  # Reset every print_every
    print_pos_acc_total = 0
    print_neg_acc_total = 0
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
        for box_feature, rank_score, box_box, box_label, box_weight, box_class_label, box_class_weight in data_loader:
        # for box_feature, rank_score, box_box, box_label in data_loader:
            start = time.time()
            # print('begin')
            box_feature_variable =  Variable(box_feature).cuda()
            box_score_variable = Variable(rank_score).cuda()
            box_label_variable = Variable(box_label).cuda()
            box_class_label_variable = Variable(box_class_label).cuda()
            # box_class_weight_variable = Variable(box_class_weight).cuda()
            box_box_variable = Variable(box_box).cuda()
            criterion_box = nn.BCELoss(weight=box_weight.cuda())
            criterion_box_class = nn.BCELoss(weight=box_class_weight.cuda())
            # criterion = nn.BCELoss(weight=box_weight.cuda())
            # criterion = nn.BCELoss(weight=box_weight.cuda())
            # print box_score_fusion_variable.size()
            loss, pos_accuracy, neg_accuracy = train(box_feature_variable, box_score_variable, box_box_variable, box_label_variable, box_class_label_variable,
                        model, model_optimizer, criterion_box, criterion_box_class)
            count += 1
            print_loss_total += loss
            print_pos_acc_total += pos_accuracy
            print_neg_acc_total += neg_accuracy
            end = time.time()
            print_time_total += float(end-start)
            if count % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_pos_acc_avg = print_pos_acc_total / print_every
                print_neg_acc_avg = print_neg_acc_total / print_every
                print_time_avg = print_time_total / print_every

                print_loss_total = 0
                print_pos_acc_total = 0
                print_neg_acc_total = 0
                print_time_total = 0
                # print('aa')
                logger.info('epoch:{}, iter:{}, lr:{}, avg_time:{:.3f}, avg_loss:{:.10f}, accuracy:{:.3f}, recall:{:.3f}'.format(epoch_index, count, learning_rate, print_time_avg, print_loss_avg, print_pos_acc_avg, print_neg_acc_avg))
            if count % save_every == 0:
                save_checkpoint(model, epoch_index, count, model_dir)

        print_loss_total = 0
        print_pos_acc_total = 0
        print_neg_acc_total = 0
        print_time_total = 0

        if epoch_index % step == 0 and epoch_index > 0:
            learning_rate = learning_rate * 0.1
            for param_group in model_optimizer.param_groups:
                param_group['lr'] = learning_rate
