import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from evaluation import COCO
from evaluation import COCOeval
# from models.rnn_model_full_global_attention_context_gate import Encoder_Decoder
from models.rnn_model_full_global_attention_context_gate_multiiou_multiious import Encoder_Decoder
# from dataset.multi_dataset_before_before_nms import TrainDataset
from dataset.multi_dataset_after_from_before_nms_threshold_v2_multiiou_multiiou import TrainDataset
# from dataset.multi_dataset_before_before_no_gt_nms import TrainDataset, unique_collate
# from dataset.multi_dataset_full_nms import TrainDataset
from solver.test_full_nms_multi_multiiou import test_solver
import torch
import argparse
import os
from multiprocessing import Process, Manager
import cvbase as cvb

def parse_args():
    parser = argparse.ArgumentParser(description='Test RNN NMS')
    parser.add_argument('--hidden_size', 
                        default=128,
                        type=int,
                        help='the hidden size of RNN')

    parser.add_argument('--gt_path', 
                        default='/data/luqi/dataset/coco/annotations/instances_val2017.json',
                        help='the path of gt json')

    parser.add_argument('--base_path', 
                        default='/data/luqi/dataset/panet_dcn/',
                        help='the data path of RNN')

    parser.add_argument('--img_list',
                        default='val.txt',
                        help='the img_list')

    parser.add_argument('--output_dir',
                        default='/data/luqi/RNN_NMS/panet_dcn_after_from_before_mlp_source_weight2_001_v2_multiiou_multiiou/',
                        help='the save path of output_dir')

    parser.add_argument('--ann_type',
                        default='bbox',
                        help='the type of anns, det or segm')

    parser.add_argument('--attn_type',
                        default='mlp',
                        help='the attn_type')

    parser.add_argument('--context_type',
                        default='source',
                        help='the attn_type, source, target, both')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    result_dir = os.path.join(args.output_dir, 'result/')
    if not osp.exists(result_dir):
        os.makedirs(result_dir)
    result_path = os.path.join(result_dir, 'result.json')
    
    model_dir = os.path.join(args.output_dir, 'model/')
    model_path = os.path.join(model_dir, 'latest.pth')
    if not osp.exists(model_path):
        raise "there is no latest.pth"

    output_dir = [model_path, result_dir]
    use_cuda = torch.cuda.is_available()

    cls_list = ['_' for _ in range(81)]
    # datasets
    # val = TrainDataset(args.base_path, args.img_list, 'msra', cls_list, phase='test')
    val = TrainDataset(args.base_path, args.img_list, 'msra', cls_list, phase='test', final_score_thresh=0.03)
    # val_loader = torch.utils.data.DataLoader(val, batch_size=1, num_workers=1, collate_fn=unique_collate, pin_memory=False)

    # model
    model = Encoder_Decoder(args.hidden_size, attn_type=args.attn_type, context_type=args.context_type)
    
    if use_cuda:
        model = model.cuda()
    model.eval()
    thread_index = 0
    thread_num = 1
    thread_result = test_solver(model, val, output_dir, thread_index, thread_num)
    # result.extend(thread_result)
    cvb.dump(thread_result, result_path)
    # do evaluation
    cocoGt = COCO(args.gt_path)
    cocoDt = cocoGt.loadRes(result_path)

    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()