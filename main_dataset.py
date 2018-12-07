#from lib.iterators.PytorchIterator import PytorchIterator
import init
import os
import sys
#sys.path.insert(0, 'lib')
from configs.faster.default_configs import config, update_config, update_config_from_list
from iterators.PytorchIterator import PytorchIterator
from data_utils.load_data import load_proposal_roidb, merge_roidb, filter_roidb
from bbox.bbox_regression import add_bbox_regression_targets
#from iterators.PytorchIterator import PytorchIterator
import torch
import argparse


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER training module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
                                                        default='configs/faster/sniper_res101_e2e.yml',type=str)
    arg_parser.add_argument('--display', dest='display', help='Number of epochs between displaying loss info',
                            default=100, type=int)
    arg_parser.add_argument('--momentum', dest='momentum', help='BN momentum', default=0.995, type=float)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()

if __name__ == '__main__':
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)
    gpu_list = [gpu for gpu in config.gpus.split(',')]
    nGPUs = len(gpu_list)
    batch_size = nGPUs * config.TRAIN.BATCH_IMAGES

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    # Create roidb
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_proposal_roidb(config.dataset.dataset, image_set, config.dataset.root_path,
        config.dataset.dataset_path,
        proposal=config.dataset.proposal, append_gt=True, flip=config.TRAIN.FLIP,
        result_path=config.output_path,
        proposal_path=config.proposal_path, load_mask=config.TRAIN.WITH_MASK, only_gt=not config.TRAIN.USE_NEG_CHIPS)
        for image_set in image_sets]

    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, config)
    bbox_means, bbox_stds = add_bbox_regression_targets(roidb, config)
    print('Creating Iterator with {} Images'.format(len(roidb)))

    pytorch_dataset = PytorchIterator(roidb=roidb, config=config, batch_size=batch_size, nGPUs=nGPUs,threads=config.TRAIN.NUM_THREAD, pad_rois_to=400)
    train_loader = torch.utils.data.DataLoader(dataset=pytorch_dataset, batch_size=batch_size, shuffle=False,num_workers=0)
    for i, (data, valid_range, im_info,label, bbox_target, bbox_weight, gt_boxes) in enumerate(train_loader):
        print(data.shape, valid_range.shape, im_info.shape,label.shape, bbox_target.shape, bbox_weight.shape, gt_boxes.shape)
        assert 1==2

    # Creating the Logger
    # logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)




