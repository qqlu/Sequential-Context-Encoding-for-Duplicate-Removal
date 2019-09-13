import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from models.rnn_model_joint import pre_post_joint
from dataset.multi_dataset_joint import TrainDataset, unique_collate
from solver.solver_full_nms_joint import solver, load_checkpoint
from solver.io import load_checkpoint_two_file
import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train RNN NMS')
    parser.add_argument('--hidden_size', 
                        default=128,
                        type=int,
                        help='the hidden size of RNN')

    parser.add_argument('--base_path', 
                        default='/mnt/lustre/liushu1/qilu_ex/dataset/coco/pytorch_data/',
                        help='the data path of RNN')

    parser.add_argument('--pre_pth', 
                        default='/mnt/lustre/liushu1/qilu_ex/RNN_NMS/mask_rcnn_before_mlp_source_weight4/model/latest.pth',
                        help='pre nms ')

    parser.add_argument('--post_pth', 
                        default='/mnt/lustre/liushu1/qilu_ex/RNN_NMS/mask_rcnn_after_from_before_mlp_source_weight2_001_v2/model/latest.pth',
                        help='post from pre nms')

    parser.add_argument('--img_list',
                        default='train.txt',
                        help='the img_list')
    
    parser.add_argument('--output_dir',
                        default='/mnt/lustre/liushu1/qilu_ex/RNN_NMS/test1/',
                        help='the save path of output_dir')

    parser.add_argument('--num_epochs',
                        default=7,
                        type=int,
                        help='the number of training epoch')

    parser.add_argument('--learning_rate',
                        default=0.01,
                        type=float,
                        help='learning_rate of model')

    parser.add_argument('--step',
                        default=5,
                        type=int,
                        help='stepsize of model')

    parser.add_argument('--load',
                        default=2,
                        type=int,
                        help='load')

    parser.add_argument('--weight',
                        default=4,
                        type=float,
                        help='weight')

    parser.add_argument('--attn_type',
                        default='mlp',
                        help='the attn_type')

    parser.add_argument('--context_type',
                        default='source',
                        help='the attn_type, source, target, both')

    parser.add_argument('--pre_post_weight',
                        default=1,
                        type=float,
                        help='pre_post_weight')

    parser.add_argument('--post_score_threshold',
                        default=0.01,
                        type=float,
                        help='post_score_threshold')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    # initialization
    model_dir = os.path.join(args.output_dir, 'model/')
    log_info_dir = os.path.join(args.output_dir, 'log/')
    load_pth = os.path.join(model_dir, 'latest.pth')

    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    if not osp.exists(log_info_dir):
        os.makedirs(log_info_dir)

    output_dir = [model_dir, log_info_dir]

    use_cuda = torch.cuda.is_available()

    # dataset
    train = TrainDataset(args.base_path, args.img_list, weight=args.weight)
    train_loader = torch.utils.data.DataLoader(train, batch_size=1, num_workers=1, collate_fn=unique_collate, pin_memory=False, shuffle=True)

    # model
    model = pre_post_joint(args.hidden_size, attn_type=args.attn_type, context_type=args.context_type, post_score_threshold=args.post_score_threshold)
    if args.load==1:
        continue_epochs, continue_iters = load_checkpoint(model, load_pth)
    elif args.load==2:
        continue_epochs, continue_iters = load_checkpoint_two_file(model, args.pre_pth, args.post_pth)

    if use_cuda:
        model = model.cuda()
    # print(model.state_dict().keys())
    if args.load==0 or args.load==2:
        solver(model, train_loader, args.num_epochs, print_every=20, save_every=10000, output_dir=output_dir, learning_rate=args.learning_rate, step=args.step, pre_post_weight=args.pre_post_weight)
    else:
        solver(model, train_loader, args.num_epochs, print_every=20, save_every=10000, output_dir=output_dir, learning_rate=args.learning_rate, step=args.step, continue_epochs=continue_epochs, continue_iters=continue_iters, pre_post_weight=args.pre_post_weight)

