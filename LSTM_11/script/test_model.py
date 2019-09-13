import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from models import Encoder_Decoder
from dataset import TrainDataset, unique_collate
from solver import test_solver
import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test RNN NMS')
    parser.add_argument('--hidden_size', 
    					default=128,
    					type=int,
    					help='the hidden size of RNN')

    parser.add_argument('--base_path', 
    					default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/pytorch_data/',
    					help='the data path of RNN')

    parser.add_argument('--img_list',
    					default='val.txt',
    					help='the img_list')
    parser.add_argument('--output_dir',
    					default='/mnt/lustre/liushu1/qilu_ex/RNN_NMS/improve_v1_sort/',
    					help='the save path of output_dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # initialization
    result_dir = os.path.join(args.output_dir, 'result/')
    model_dir = os.path.join(args.output_dir, 'model/')
    model_path = os.path.join(model_dir, 'latest.pth')
    if not osp.exists(model_path):
        raise "there is no latest.pth "
    if not osp.exists(result_dir):
        os.makedirs(result_dir)

    output_dir = [model_path, result_dir]
    use_cuda = torch.cuda.is_available()

    cls_list = ['_' for _ in range(81)]
    # datasettes
    train = TrainDataset(args.base_path, args.img_list, 'msra', cls_list, phase='train')
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, num_workers=1, collate_fn=unique_collate, pin_memory=False)

    # model
    model = Encoder_Decoder(args.hidden_size)

    if use_cuda:
        model = model.cuda()

    for box_feature, rank_score, box_box, box_label in train_loader:
        # print('main:', np.sum(box_label.numpy())
        # input()