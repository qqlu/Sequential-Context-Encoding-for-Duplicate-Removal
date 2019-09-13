import sys
# sys.path.append('/data/code/qilu_tools/gpu')
#from gpu_nms import gpu_nms
import numpy as np
# import cv2
import time
import cvbase as cvb
import os
import pickle as pkl

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        content = f.readlines()
        all_num = len(content)
        single_num = int(all_num / 3)
        box_feature_content = content[0:single_num]
        box_box_content = content[single_num*2:single_num*3]
        box_score_content = content[single_num:single_num*2]

        box_feature = np.zeros((single_num, 1024), dtype=np.float16)
        box_box = np.zeros((single_num, 324), dtype=np.float16)
        box_score = np.zeros((single_num,  81), dtype=np.float16)

        for row_index in range(single_num):
            row_content = box_feature_content[row_index].strip('\n').strip(' ').split(' ')
            assert len(row_content) == 1024, 'row:{}'.format(len(row_content))
            for col_index in range(1024):
                box_feature[row_index][col_index] = row_content[col_index]

        for row_index in range(single_num):
            row_content = box_box_content[row_index].strip('\n').strip(' ').split(' ')
            assert len(row_content) == 324, len(row_content)
            for col_index in range(324):
                box_box[row_index][col_index] = row_content[col_index]

        for row_index in range(single_num):
            row_content = box_score_content[row_index].strip('\n').strip(' ').split(' ')
            assert len(row_content) == 81
            for col_index in range(81):
                box_score[row_index][col_index] = row_content[col_index]

    return box_feature, box_box, box_score

def transForm(box_feature, box_box, box_score, save_path):
    cvb.dump([box_feature, box_box, box_score], save_path)


def visualize_from_txt(img_path, box_box, box_score):
    I = cv2.imread(img_path)
    class_box_list = []
    # img_path = '/data/dataset/coco_dataset/train2017/000000201270.jpg'
    for cls_index in range(1, 81):
        temp = np.concatenate((box_box[:, cls_index*4:cls_index*4+4], box_score[:, cls_index:cls_index+1]), 1)
        class_box_list.append(temp)

    class_vis_list = []
    for cls_info in class_box_list:
        keep = gpu_nms(cls_info, 0.3, device_id=0)
        cls_info_info = cls_info[keep, :]
        class_vis_list.append(cls_info_info)
    for cls_info in class_vis_list:
        for index in range(cls_info.shape[0]):
            if cls_info[index, 4]>0.5:
                cv2.rectangle(I, (cls_info[index, 0], cls_info[index, 1]), (cls_info[index, 2], cls_info[index, 3]), (0, 255, 0), 1)
    cv2.namedWindow('bb')
    cv2.imshow('bb', I)
    cv2.waitKey(0)

def visualize_from_txt_without_nms(img_path, box_box, box_score):
    I = cv2.imread(img_path)
    class_box_list = []
    # img_path = '/data/dataset/coco_dataset/train2017/000000201270.jpg'
    for cls_index in range(1, 81):
        temp = np.concatenate((box_box[:, cls_index*4:cls_index*4+4], box_score[:, cls_index:cls_index+1]), 1)
        class_box_list.append(temp)

    class_vis_list = []
    for cls_info in class_box_list:
        class_vis_list.append(cls_info)

    for cls_info in class_vis_list:
        for index in range(cls_info.shape[0]):
            if cls_info[index, 4]>0.5:
                cv2.rectangle(I, (cls_info[index, 0], cls_info[index, 1]), (cls_info[index, 2], cls_info[index, 3]), (0, 255, 0), 1)
    cv2.namedWindow('bb')
    cv2.imshow('bb', I)
    cv2.waitKey(0)

if __name__ == '__main__':
    txt_path = '/mnt/lustre/liushu1/qilu_ex/dataset/test_dev/panet/test.txt'
    info_path = '/mnt/lustre/liushu1/qilu_ex/rnn_nms/panet_dcn/testfolder/test16w_new_dev/output/'
    save_path = '/mnt/lustre/liushu1/qilu_ex/dataset/test_dev/panet/info/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    thread_num = 16
    thread_idx = int(sys.argv[1])

    img_list = cvb.list_from_file(txt_path)
    num_list = len(img_list)
    for idx in range(0, num_list):
        if idx % thread_num != thread_idx:
            continue
        start = time.time()
        save_pkl_path = os.path.join(save_path, img_list[idx] + '.pkl')
        
        info_txt_path = os.path.join(info_path, 'COCO_train2014_' + img_list[idx] + '.txt')
        if not os.path.exists(info_txt_path):   
            info_txt_path = os.path.join(info_path, 'COCO_val2014_' + img_list[idx] + '.txt')
            if not os.path.exists(info_txt_path):
                info_txt_path = os.path.join(info_path, 'COCO_test2015_' + img_list[idx] + '.txt')
                if not os.path.exists(info_txt_path):
                    continue
        box_feature, box_box, box_score = read_txt(info_txt_path)
        transForm(box_feature, box_box, box_score, save_pkl_path)
        end = time.time()
        print('{}/{}, thread_idx:{}, img_idx:{}, cost:{:.3f}'.format(idx, num_list, thread_idx, img_list[idx], float(end-start)))
    #I = cv2.imread(img_path)
    # start = time.time()
    # for i in range(0, 100):
    #     print(i)
    #     box_feature, box_box, box_score = cvb.load(save_path)
    # end = time.time()
    # print(float(end-start)/100.0)
    # transForm(box_feature, box_box, box_score, save_path)
    

