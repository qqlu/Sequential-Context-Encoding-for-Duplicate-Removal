import numpy as np
import torch
from gpu.cpu_nms import cpu_nms
from gpu.cpu_nms_cython import nms
from gpu.soft_cpu_nms import cpu_soft_nms
import sys
sys.path.append('')

def bbox_overlaps(bboxes1, bboxes2):
    """Calculate the ious between each bbox of bboxes1 and bboxes2
    Args:
        bboxes1(numpy): shape (n, 4)
        bboxes2(numpy): shape (k, 4)
    Returns:
        ious(numpy): shape (n, k)
    """
    N = bboxes1.shape[0]
    K = bboxes2.shape[0]
    overlaps = np.zeros((N, K)).astype(np.float)
    if N * K is 0:
        return overlaps
    
    for k in range(K):    
        box_area = (bboxes2[k, 2] - bboxes2[k, 0] + 1) * (bboxes2[k, 3] - bboxes2[k, 1] + 1)
        for n in range(N):
            iw = min(bboxes1[n, 2], bboxes2[k, 2]) - max(bboxes1[n, 0], bboxes2[k, 0]) + 1
            if iw > 0:
                ih = min(bboxes1[n, 3], bboxes2[k, 3]) - max(bboxes1[n, 1], bboxes2[k, 1]) + 1
                if ih > 0:
                    ua = (bboxes1[n, 2] - bboxes1[n, 0] + 1) * (bboxes1[n, 3] - bboxes1[n, 1] + 1) + box_area - iw * ih
                    overlaps[n, k] = iw * ih / ua

    return overlaps


def assign_label(proposals,
                gt_bboxes,
                pos_iou_thr=0.5,
                weight=None, 
                ):

    num_proposals = proposals.shape[0]
    num_gts = gt_bboxes.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    proposals_weight = np.zeros((num_proposals, 1))

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    for unique_gt in unique_gts:
        if unique_gt>=100:
            unique_gt = unique_gt - 100
        cls_proposals_box = proposals[:,unique_gt*4:unique_gt*4+4]
        index = np.where(gt_bboxes[:, 4]==unique_gt)[0]

        cls_gts_box = gt_bboxes[index, 0:4]
        #print('cls_proposals_box.shape:{}, cls_gts_box.shape:{}'.format(cls_proposals_box.shape, cls_gts_box.shape))
        ious = bbox_overlaps(cls_proposals_box, cls_gts_box)
        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                proposals_label[argmax_overlaps[index]] = 1
                proposals_weight[argmax_overlaps[index]] = weight

    if weight is None:
        return proposals_label
    else:
        return proposals_label, proposals_weight

def assign_label_with_nms(proposals,
                           proposals_score,
                           iou_thr=0.3,
                           weight=None,
                           score_thr=0.01):

    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    proposals_weight = np.zeros((num_proposals, 1))
    class_box_list = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        cls_info = cls_info[valid_index, :]
        
        keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        proposals_label[valid_index[keep], :] = 1
    
    index = np.where(proposals_label==1)[0]
    proposals_weight[index, :] = weight
    if weight is None:
        return proposals_label
    else:
        return proposals_label, proposals_weight

def verify_nms(proposals,
               proposals_score,
               image_id,
               New2Old,
               iou_thr=0.3,
               score_thr=0.01):
    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    bboxes = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        cls_info = cls_info[valid_index, :]

        # keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        keep = nms(cls_info.astype(np.float32), iou_thr)
        for index in list(keep):
            x1, y1, x2, y2, score = cls_info[index, :]
            # category_id = New2Old[str(cls_index)][1]
            category_id = cls_index
            # bboxes.append({'bbox': [float(x1), float(y1), float(x2)-float(x1)+1, float(y2)-float(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
    # print len(bboxes)
    # raw_input()
    return bboxes

def verify_nms_with_limit(proposals,
               proposals_score,
               image_id,
               New2Old,
               iou_thr=0.3,
               score_thr=0.01):
    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    bboxes = []
    cls_boxes = []
    for cls_index in range(1, 81):
        # cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        cls_info = np.hstack((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1])).astype(np.float32, copy=False)
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        cls_info = cls_info[valid_index, :]
    
        # keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        keep = nms(cls_info.astype(np.float32), iou_thr)
        cls_boxes.append(cls_info[keep, :])

    image_scores = np.hstack([cls_boxes[j][:, -1] for j in range(80)])
    
    if len(image_scores) > 100:
        image_thresh = np.sort(image_scores)[-100]
    else:
        image_thresh = 0

    for cls_index in range(80):
        cls_info = cls_boxes[cls_index]
        keep = np.arange(cls_info.shape[0])
        for index in list(keep):
            x1, y1, x2, y2, score = cls_info[index, :]
            if score<image_thresh:
                continue
            category_id = New2Old[str(cls_index+1)][1]
            bboxes.append({'bbox': [float(x1), float(y1), float(x2)-float(x1)+1, float(y2)-float(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})

    # print len(bboxes)
    # raw_input()
    return bboxes

def verify_soft_nms_with_limit(proposals,
               proposals_score,
               image_id,
               New2Old,
               iou_thr=0.3):
    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    bboxes = []
    cls_boxes = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if len(valid_index)==0:
            continue
        cls_info = cls_info[valid_index, :].astype(np.float32)
        keep = cpu_soft_nms(cls_info, method=2)

        for index in list(keep):
            x1, y1, x2, y2, score = cls_info[index, :]
            category_id = New2Old[str(cls_index)][1]
            bboxes.append({'bbox': [float(x1), float(y1), float(x2)-float(x1)+1, float(y2)-float(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
    return bboxes

def verify_soft_nms(proposals,
               proposals_score,
               image_id,
               New2Old,
               iou_thr=0.3,
               score_thr=0.01):
    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    bboxes = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        if len(valid_index)==0:
            continue
        cls_info = cls_info[valid_index, :].astype(np.float32)
        keep = cpu_soft_nms(cls_info, method=2)
        # print(len(keep))
        # print(cls_info.shape[0])
        for index in list(keep):
            x1, y1, x2, y2, score = cls_info[index, :]
            category_id = New2Old[str(cls_index)][1]
            # bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})   
    return bboxes

def assign_label_80_with_nms(proposals,
                           proposals_score,
                           iou_thr=0.3,
                           weight=None,
                           score_thr=0.01):

    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    proposals_weight = np.ones((num_proposals, 1))
    class_box_list = []
    for cls_index in range(1, 81):
        cls_index -= 1
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        cls_info = cls_info[valid_index, :]
        
        keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        proposals_label[valid_index[keep], :] = 1
    
    index = np.where(proposals_label==1)[0]
    proposals_weight[index, :] = weight
    if weight is None:
        return proposals_label
    else:
        return proposals_label, proposals_weight

def assign_label_80_with_nms_multi(proposals,
                           proposals_score,
                           iou_thr=0.3,
                           weight=None,
                           score_thr=0.01):

    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.zeros((num_proposals, 80))
    class_box_list = []
    valid_count = 0.
    for cls_index in range(1, 81):
        cls_index -= 1
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        cls_info = cls_info[valid_index, :]
        
        keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        valid_count += len(keep)
        proposals_label[valid_index[keep], cls_index] = 1
    weight = (num_proposals*80-valid_count) / valid_count
    # print('weight:{}'.format(weight)) 
    row_index, col_index = np.where(proposals_label==1)
    proposals_weight[row_index, col_index] = int(np.log(weight))
    if weight is None:
        return proposals_label
    else:
        return proposals_label, proposals_weight


def stage2_filter(proposals,
               proposals_score,
               gt_bboxes,
               image_id,
               New2Old,
               iou_thr=0.3,
               pos_iou_thr=0.5):

    num_proposals = proposals.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if len(valid_index)==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[keep, :]

        # gts
        valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[valid_index, 0:4]
        #print('cls_proposals_box.shape:{}, cls_gts_box.shape:{}'.format(cls_proposals_box.shape, cls_gts_box.shape))
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)
        # print(ious.shape)
        # input()
        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                x1, y1, x2, y2, score = cls_nms_info[arg_index, :]
                category_id = New2Old[str(cls_index)][1]
                bboxes.append({'bbox': [int(x1), int(y1), int(x2)-int(x1)+1, int(y2)-int(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
    
    return bboxes

def stage2_assign(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                nms_iou_thr=0.3,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    keep_all_cls = []

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), nms_iou_thr)
        keep_all_cls.extend(valid_index[keep])

    keep_all_cls = np.unique(np.array(keep_all_cls))
    proposals_feature_nms = proposals_feature[keep_all_cls, :]
    proposals_score_nms = proposals_score[keep_all_cls, :]
    proposals_box_nms = proposals_box[keep_all_cls, :]
    proposals_label = np.zeros((keep_all_cls.shape[0], 1))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box_nms[:, cls_index*4:cls_index*4+4], proposals_score_nms[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]]] = 1

    return proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label

def stage2_assign_weight(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                nms_iou_thr=0.3,
                pos_iou_thr=0.5,):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    keep_all_cls = []

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), nms_iou_thr)
        keep_all_cls.extend(valid_index[keep])

    keep_all_cls = np.unique(np.array(keep_all_cls))
    proposals_feature_nms = proposals_feature[keep_all_cls, :]
    proposals_score_nms = proposals_score[keep_all_cls, :]
    proposals_box_nms = proposals_box[keep_all_cls, :]
    proposals_label = np.zeros((keep_all_cls.shape[0], 1))
    proposals_weight = np.ones((keep_all_cls.shape[0], 1))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box_nms[:, cls_index*4:cls_index*4+4], proposals_score_nms[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]]] = 1
                proposals_weight[valid_index[keep[arg_index]]] = weight

    return proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_weight


def stage2_assign_weight_slack(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5,):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    keep_all_cls = []
    keep_all_np = np.zeros((proposals_score.shape[0], 80))

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), nms_iou_thr)
        keep_all_np[valid_index[keep], cls_index-1] = 1
        keep_all_cls.extend(valid_index[keep])

    keep_all_cls = np.unique(np.array(keep_all_cls))
    proposals_feature_nms = proposals_feature[keep_all_cls, :]
    proposals_score_nms = proposals_score[keep_all_cls, :]
    proposals_box_nms = proposals_box[keep_all_cls, :]
    proposals_keep_np = keep_all_np[keep_all_cls, :]
    proposals_label = np.zeros((keep_all_cls.shape[0], 1))
    proposals_weight = np.ones((keep_all_cls.shape[0], 1))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box_nms[:, cls_index*4:cls_index*4+4], proposals_score_nms[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]]] = 1
                proposals_weight[valid_index[keep[arg_index]]] = weight

    return proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_weight, proposals_keep_np

def stage2_assign_weight_slack_has_class(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    keep_all_cls = []
    keep_all_np = np.zeros((proposals_score.shape[0], 80))

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), nms_iou_thr)
        keep_all_np[valid_index[keep], cls_index-1] = 1
        keep_all_cls.extend(valid_index[keep])

    keep_all_cls = np.unique(np.array(keep_all_cls))
    proposals_feature_nms = proposals_feature[keep_all_cls, :]
    proposals_score_nms = proposals_score[keep_all_cls, :]
    proposals_box_nms = proposals_box[keep_all_cls, :]
    proposals_keep_np = keep_all_np[keep_all_cls, :]
    proposals_label = np.zeros((keep_all_cls.shape[0], 1))
    proposals_weight = np.ones((keep_all_cls.shape[0], 1))
    proposals_class = np.zeros((keep_all_cls.shape[0], 1))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box_nms[:, cls_index*4:cls_index*4+4], proposals_score_nms[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]]] = 1
                proposals_weight[valid_index[keep[arg_index]]] = weight
                proposals_class[valid_index[keep[arg_index]]] = cls_index
    return proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_weight, proposals_class, proposals_keep_np

def stage2_assign_weight_slack_soft(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                pos_iou_thr=0.5,
                method=1):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    keep_all_cls = []
    keep_all_np = np.zeros((proposals_score.shape[0], 80))

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :].astype(np.float32)
        keep = cpu_soft_nms(cls_info, method=method)
        keep_all_np[valid_index[keep], cls_index-1] = 1
        keep_all_cls.extend(valid_index[keep])

    keep_all_cls = np.unique(np.array(keep_all_cls))
    proposals_feature_nms = proposals_feature[keep_all_cls, :]
    proposals_score_nms = proposals_score[keep_all_cls, :]
    proposals_box_nms = proposals_box[keep_all_cls, :]
    proposals_keep_np = keep_all_np[keep_all_cls, :]
    proposals_label = np.zeros((keep_all_cls.shape[0], 1))
    proposals_weight = np.ones((keep_all_cls.shape[0], 1))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box_nms[:, cls_index*4:cls_index*4+4], proposals_score_nms[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_soft_nms(cls_info_first.astype(np.float32), method=method)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]]] = 1
                proposals_weight[valid_index[keep[arg_index]]] = weight

    return proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_weight, proposals_keep_np

def stage2_assign_weight_single_class(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=2,
                nms_iou_thr=0.3,
                pos_iou_thr=0.5,):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)
    bboxes = []
    keep_all_cls = []
    keep_all_np = np.zeros((proposals_score.shape[0], 80))

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), nms_iou_thr)
        keep_all_np[valid_index[keep], cls_index-1] = 1
        keep_all_cls.extend(valid_index[keep])

    keep_all_cls = np.unique(np.array(keep_all_cls))
    proposals_feature_nms = proposals_feature[keep_all_cls, :]
    proposals_score_nms = proposals_score[keep_all_cls, :]
    proposals_box_nms = proposals_box[keep_all_cls, :]
    proposals_keep_np = keep_all_np[keep_all_cls, :]
    proposals_label = np.zeros((keep_all_cls.shape[0], 1))
    proposals_weight = np.ones((keep_all_cls.shape[0], 1))
    proposals_class_label = np.zeros((keep_all_cls.shape[0], 1))
    proposals_class_weight = np.ones((keep_all_cls.shape[0], 1))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box_nms[:, cls_index*4:cls_index*4+4], proposals_score_nms[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]
        # class_label
        proposals_class_label[valid_index[keep], 0] = 1
        proposals_class_weight[valid_index[keep], 0] = weight
        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]]] = 1
                proposals_weight[valid_index[keep[arg_index]]] = weight

    return proposals_feature_nms, proposals_score_nms, proposals_box_nms, proposals_label, proposals_class_label, proposals_weight, proposals_class_weight, proposals_keep_np

def stage_full_assign_weight_slack_has_class(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)

    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.ones((num_proposals, 80))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[keep[arg_index]], cls_index-1] = 1
                proposals_weight[valid_index[keep[arg_index]], cls_index-1] = weight
                # proposals_class[valid_index[keep[arg_index]]] = cls_index
                # proposals_keep_np[valid_index[keep[arg_index]], cls_index] = 1

    return proposals_label, proposals_weight

def stage1_before_assign_weight_slack_has_class(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)

    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.ones((num_proposals, 80))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        proposals_label[valid_index[keep], cls_index-1] = 1
        proposals_weight[valid_index[keep], cls_index-1] = weight
        # proposals_class[valid_index[keep[arg_index]]] = cls_index
        # proposals_keep_np[valid_index[keep[arg_index]], cls_index] = 1

    return proposals_label, proposals_weight

def stage1_before_no_gt_assign_weight_slack_has_class(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    # unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)

    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.ones((num_proposals, 80))

    for cls_index in range(1, 81):
        # if cls_index>=100:
        #     continue
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        proposals_label[valid_index[keep], cls_index-1] = 1
        proposals_weight[valid_index[keep], cls_index-1] = weight
        # proposals_class[valid_index[keep[arg_index]]] = cls_index
        # proposals_keep_np[valid_index[keep[arg_index]], cls_index] = 1

    return proposals_label, proposals_weight

def stage2_nms_proposals(proposals_box,
                proposals_score,
                score_thresh=0.01,
                nms_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    keep_nms_np = np.zeros((num_proposals, 80))

    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > 0.01)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info = cls_info[valid_index, :]
        keep = cpu_nms(cls_info.astype(np.float32), nms_iou_thr)
        keep_nms_np[valid_index[keep], cls_index-1] = 1

    return keep_nms_np

def stage_full_assign_weight_slack_has_class_v3(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)

    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.ones((num_proposals, 80))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_info_first = cls_info[valid_index, :]

        keep = cpu_nms(cls_info_first.astype(np.float32), nms_iou_thr)
        # proposal after nms
        cls_nms_info = cls_info[valid_index[keep], :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_nms_info[:,0:4], cls_gts_box)

        for col_index in range(ious.shape[1]):
            row_satis = np.where(ious[:, col_index]>=pos_iou_thr)[0]
            if len(row_satis)==0:
                continue
            argmax_satis = np.argmax(cls_nms_info[row_satis, 4], axis=0)
            proposals_label[valid_index[keep[row_satis[argmax_satis]]], cls_index-1] = 1
            proposals_weight[valid_index[keep[row_satis[argmax_satis]]], cls_index-1] = weight

    return proposals_label, proposals_weight

def stage_full_assign_weight_slack_has_class_v4(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)

    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.ones((num_proposals, 80))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_valid_info = cls_info[valid_index, :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_valid_info[:,0:4], cls_gts_box)

        max_overlaps = np.max(ious, axis=0)
        argmax_overlaps= np.argmax(ious, axis=0)
        
        for index in range(argmax_overlaps.shape[0]):
            if max_overlaps[index] > pos_iou_thr:
                arg_index = argmax_overlaps[index]
                proposals_label[valid_index[arg_index], cls_index-1] = 1
                proposals_weight[valid_index[arg_index], cls_index-1] = weight

        # for col_index in range(ious.shape[1]):
        #     row_satis = np.where(ious[:, col_index]>=pos_iou_thr)[0]
        #     if len(row_satis)==0:
        #         continue
        #     argmax_satis = np.argmax(cls_nms_info[row_satis, 4], axis=0)
        #     proposals_label[valid_index[row_satis[argmax_satis]], cls_index-1] = 1
        #     proposals_weight[valid_index[row_satis[argmax_satis]], cls_index-1] = weight
            # proposals_label[start+row_satis[argmax_satis], 0] = 1
            # proposals_weight[start+row_satis[argmax_satis], 0] = weight

        # max_overlaps = np.max(ious, axis=0)
        # argmax_overlaps= np.argmax(ious, axis=0)
        
        # for index in range(argmax_overlaps.shape[0]):
        #     if max_overlaps[index] > pos_iou_thr:
        #         arg_index = argmax_overlaps[index]
        #         proposals_label[valid_index[keep[arg_index]], cls_index-1] = 1
        #         proposals_weight[valid_index[keep[arg_index]], cls_index-1] = weight
                # proposals_class[valid_index[keep[arg_index]]] = cls_index
                # proposals_keep_np[valid_index[keep[arg_index]], cls_index] = 1

    return proposals_label, proposals_weight

def stage_full_assign_weight_slack_has_class_v5(proposals_feature,
                proposals_box,
                proposals_score,
                gt_bboxes,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):

    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    unique_gts = np.unique(gt_bboxes[:, 4]).astype(np.int16)

    proposals_label = np.zeros((num_proposals, 80))
    proposals_weight = np.ones((num_proposals, 80))

    for cls_index in unique_gts:
        if cls_index>=100:
            continue
        cls_info = np.concatenate((proposals_box[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
        if valid_index.shape[0]==0:
            continue
        cls_valid_info = cls_info[valid_index, :]

        # gts
        gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
        cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
        ious = bbox_overlaps(cls_valid_info[:,0:4], cls_gts_box)


        for col_index in range(ious.shape[1]):
            row_satis = np.where(ious[:, col_index]>=pos_iou_thr)[0]
            if len(row_satis)==0:
                continue
            argmax_satis = np.argmax(cls_valid_info[row_satis, 4], axis=0)
            proposals_label[valid_index[row_satis[argmax_satis]], cls_index-1] = 1
            proposals_weight[valid_index[row_satis[argmax_satis]], cls_index-1] = weight

    return proposals_label, proposals_weight

def stage_full_assign_weight_slack_has_class_v6(proposals_box,
                proposals_score,
                gt_bboxes,
                cls_index,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5,
                pos_iou_thr=0.5):
    
    cls_index += 1
    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    proposals_label = np.zeros((num_proposals, 1))
    proposals_weight = np.ones((num_proposals, 1))

   
    cls_info = np.concatenate((proposals_box[:, :], proposals_score[:, 0:1]), 1)
    valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
    if valid_index.shape[0]==0:
        return proposals_label, proposals_weight
    cls_valid_info = cls_info[valid_index, :]

    # gts
    gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
    cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
    ious = bbox_overlaps(cls_valid_info[:,0:4], cls_gts_box)


    for col_index in range(ious.shape[1]):
        row_satis = np.where(ious[:, col_index]>=pos_iou_thr)[0]
        if len(row_satis)==0:
            continue
        argmax_satis = np.argmax(cls_valid_info[row_satis, 4], axis=0)
        proposals_label[valid_index[row_satis[argmax_satis]], 0] = 1
        proposals_weight[valid_index[row_satis[argmax_satis]], 0] = weight

    return proposals_label, proposals_weight

def stage_full_assign_weight_slack_has_class_v6_multiiou(proposals_box,
                proposals_score,
                gt_bboxes,
                cls_index,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5):
    
    cls_index += 1
    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    proposals_label = np.zeros((num_proposals, 5))
    proposals_weight = np.ones((num_proposals, 5))

   
    cls_info = np.concatenate((proposals_box[:, :], proposals_score[:, 0:1]), 1)
    valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
    if valid_index.shape[0]==0:
        return proposals_label, proposals_weight
    cls_valid_info = cls_info[valid_index, :]

    # gts
    gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
    cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
    ious = bbox_overlaps(cls_valid_info[:,0:4], cls_gts_box)

    # multiiou
    pos_iou_thr_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    for col_index in range(ious.shape[1]):
        for iou_index, pos_iou_thr in enumerate(pos_iou_thr_list):
            row_satis = np.where(ious[:, col_index]>=pos_iou_thr)[0]
            if len(row_satis)==0:
                continue
            argmax_satis = np.argmax(cls_valid_info[row_satis, 4], axis=0)
            proposals_label[valid_index[row_satis[argmax_satis]], iou_index] = 1
            proposals_weight[valid_index[row_satis[argmax_satis]], iou_index] = weight+iou_index

    return proposals_label, proposals_weight

def stage_full_assign_weight_slack_has_class_v6_multiiou_multiiou(proposals_box,
                proposals_score,
                gt_bboxes,
                cls_index,
                weight=1,
                score_thresh=0.01,
                nms_iou_thr=0.5):
    
    cls_index += 1
    num_proposals = proposals_box.shape[0]
    num_gts = gt_bboxes.shape[0]

    proposals_label = np.zeros((num_proposals, 10))
    proposals_weight = np.ones((num_proposals, 10))

   
    cls_info = np.concatenate((proposals_box[:, :], proposals_score[:, 0:1]), 1)
    valid_index = np.where(cls_info[:, -1] > score_thresh)[0]
    if valid_index.shape[0]==0:
        return proposals_label, proposals_weight
    cls_valid_info = cls_info[valid_index, :]

    # gts
    gt_valid_index = np.where(gt_bboxes[:, 4]==cls_index)[0]
    cls_gts_box = gt_bboxes[gt_valid_index, 0:4]
    ious = bbox_overlaps(cls_valid_info[:,0:4], cls_gts_box)

    # multiiou
    pos_iou_thr_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aa = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    for col_index in range(ious.shape[1]):
        for iou_index, pos_iou_thr in enumerate(pos_iou_thr_list):
            row_satis = np.where(ious[:, col_index]>=pos_iou_thr)[0]
            if len(row_satis)==0:
                continue
            argmax_satis = np.argmax(cls_valid_info[row_satis, 4], axis=0)
            proposals_label[valid_index[row_satis[argmax_satis]], iou_index] = 1
            proposals_weight[valid_index[row_satis[argmax_satis]], iou_index] = weight + aa[iou_index]

    return proposals_label, proposals_weight

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    
    # if(inter.size(0)>=15):
    #     print(box_a)
    #     print(box_b)
    #     print(area_a[15,15], area_b[15,15], inter[15,15])
    #     input()
    union = area_a + area_b - inter + 1e-10
    
    return inter / union  # [A,B]


def verify_nms_with_box_voting(proposals,
               proposals_score,
               image_id,
               New2Old,
               iou_thr=0.3,
               score_thr=0.01,
               bv_method='ID'):
    num_proposals = proposals.shape[0]
    proposals_label = np.zeros((num_proposals, 1))
    bboxes = []
    for cls_index in range(1, 81):
        cls_info = np.concatenate((proposals[:, cls_index*4:cls_index*4+4], proposals_score[:, cls_index:cls_index+1]), 1)
        valid_index = np.where(cls_info[:, -1] > score_thr)[0]
        cls_info = cls_info[valid_index, :]

        # keep = cpu_nms(cls_info.astype(np.float32), iou_thr)
        keep = nms(cls_info.astype(np.float32), iou_thr)
        cls_nms_info = cls_info[keep, :]
        num_dets = box_voting(cls_nms_info, cls_info, 0.8, scoring_method=bv_method)
        for index in range(num_dets.shape[0]):
            x1, y1, x2, y2, score = num_dets[index, :]
            category_id = New2Old[str(cls_index)][1]
            # bboxes.append({'bbox': [float(x1), float(y1), float(x2)-float(x1)+1, float(y2)-float(y1)+1], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
            bboxes.append({'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)], 'score': float(score), 'category_id':category_id, 'image_id':int(image_id)})
    # print len(bboxes)
    # raw_input()
    return bboxes

def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out