import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from utils import visualize_from_log
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train RNN NMS')
    parser.add_argument('--output_dir',
    					default='/home/qqlu/temp/log/test.log',
    					help='the save path of output_dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # initialization
    log_info_dir = args.output_dir
    visualize_from_log(log_info_dir)
