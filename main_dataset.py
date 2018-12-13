
import init
import os
import sys
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib
matplotlib.use('Agg')
#sys.path.insert(0, 'lib')
from configs.faster.default_configs import config, update_config, update_config_from_list
from iterators.PytorchIterator import PytorchIterator
from data_utils.load_data import load_proposal_roidb, merge_roidb, filter_roidb
from bbox.bbox_regression import add_bbox_regression_targets
from iterators.PytorchIterator import PytorchIterator
from models.faster_rcnn import FasterRCNN
from train_utils.train_one_batch import train_one_batch
import argparse
import logging
import logging.config

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 0 if (self.count < 1e-5) else (self.sum / self.count)





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

def save_checkpoint(state, filename):
    torch.save(state, filename)

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

    # Creating the Logger


    pytorch_dataset = PytorchIterator(roidb=roidb, config=config, batch_size=batch_size, nGPUs=nGPUs,threads=config.TRAIN.NUM_THREAD, pad_rois_to=400)
    train_loader = torch.utils.data.DataLoader(dataset=pytorch_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

    train_model = FasterRCNN(config,is_train=True)


    train_model = torch.nn.DataParallel(train_model).cuda().train()

    optimizer = torch.optim.SGD(train_model.parameters(), 0.01, momentum=0.9, weight_decay=0.0001)
    #train_model = train_model.cuda()
    meter_names = ['batch_time',  'loss','rpn_cls_loss','rpn_box_loss','rcnn_cls_loss','rcnn_box_loss','acc','pos_recall','neg_recall','neg_num','pos_num','rpn_acc','rpn_pos_recall','rpn_neg_recall','rpn_neg_num','rpn_pos_num']
    meters = {name: AverageMeter() for name in meter_names}
    for epoch in range(config.TRAIN.begin_epoch,config.TRAIN.end_epoch):
        for i, (data, valid_range, im_info,label, bbox_target, bbox_weight, gt_boxes) in enumerate(train_loader):

            #print(data.shape, valid_range.shape, im_info.shape,label.shape, bbox_target.shape, bbox_weight.shape, gt_boxes.shape)
            train_one_batch(train_model,optimizer,meters,data, valid_range, im_info,label, bbox_target, bbox_weight, gt_boxes,epoch,i)

        if epoch!=0 and epoch%2==0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.1

        save_name = os.path.join('/home/liuqiuyue/snipper_pytorch/output', 'faster_rcnn_{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': train_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name)






