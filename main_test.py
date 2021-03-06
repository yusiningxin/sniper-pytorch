import init
import os
import sys
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
#sys.path.insert(0, 'lib')
from configs.faster.default_configs import config, update_config, update_config_from_list
from iterators.PytorchIterator import PytorchIterator
from data_utils.load_data import load_proposal_roidb, merge_roidb, filter_roidb
from iterators.PytorchIterator import PytorchIterator
from model.faster_rcnn.resnet import resnet

import argparse
from inference import imdb_detection_wrapper
import logging
import math

import logging.config


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER test module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--vis', dest='vis', help='Whether to visualize the detections',
                            action='store_true')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()



def main():
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Create roidb
    #print(config)

    roidb, imdb = load_proposal_roidb(config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path,
                                      config.dataset.dataset_path,
                                      proposal=config.dataset.proposal, only_gt=True, flip=False,
                                      result_path=config.output_path,
                                      proposal_path=config.proposal_path, get_imdb=True)
    #roidb = roidb[:100]
    #check_point = torch.load('output/faster_rcnn_1220_0_19000.pth')

    load_name = 'output/nofix_3_15000.pth'
    #load_name = 'output/faster_rcnn_jwyang.pth'

    # faster-rcnn
    fasterRCNN = resnet(config.dataset.NUM_CLASSES, 101, pretrained=True, class_agnostic=config.CLASS_AGNOSTIC)
    # init weight
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k[0:6] == 'module':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    fasterRCNN.load_state_dict(new_state_dict)

    fasterRCNN.cuda()
    fasterRCNN = nn.DataParallel(fasterRCNN)
    fasterRCNN.eval()
    if config.TEST.EXTRACT_PROPOSALS:
        imdb_proposal_extraction_wrapper(sym_inst, config, imdb, roidb, context, arg_params, aux_params, args.vis)
    else:
        imdb_detection_wrapper(fasterRCNN, config, imdb, roidb)

if __name__ == '__main__':
    main()



