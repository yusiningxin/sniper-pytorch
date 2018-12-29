
import init
import os
import sys
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
nGPUs = 3
import matplotlib
matplotlib.use('Agg')
import torch.nn as nn
from configs.faster.default_configs import config, update_config, update_config_from_list
from iterators.PytorchIterator import PytorchIterator
from data_utils.load_data import load_proposal_roidb, merge_roidb, filter_roidb
from bbox.bbox_regression import add_bbox_regression_targets
from iterators.PytorchIterator import PytorchIterator
import argparse
import logging
import math
import logging.config
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient
import time
import numpy

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


def pos_neg_recall(output, target):
    output = numpy.float32(output.data.cpu().numpy())
    pred = output.argmax(axis=1)
    label = numpy.int32(target.data.cpu().numpy())
    correct = (pred == label)
    neg_label = (label == 0)
    neg_num = neg_label.sum()
    neg_recall_num = numpy.sum(correct * neg_label)
    pos_label = (label > 0)
    pos_num = pos_label.sum()
    pos_recall_num = numpy.sum(correct * pos_label)
    correct_num = numpy.sum(correct)
    pos_recall = pos_recall_num*1.0/pos_num if pos_num!=0 else 0
    neg_recall = neg_recall_num*1.0/neg_num if neg_num!=0 else 0
    acc = correct_num*1.0/(pos_num+neg_num)
    return pos_recall, neg_recall,acc,pos_num,neg_num


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


    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    valid_range = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    valid_range = valid_range.cuda()
    im_info = im_info.cuda()
    gt_boxes = gt_boxes.cuda()

    #faster-rcnn
    fasterRCNN = resnet(config.dataset.NUM_CLASSES, 101, pretrained=True, class_agnostic=config.CLASS_AGNOSTIC)
    # init weight
    fasterRCNN.create_architecture()

    lr = 0.001

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    load_name = 'output/faster_rcnn_jwyang.pth'
    checkpoint = torch.load(load_name)
    origin_state_dict = fasterRCNN.state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        if k[0:6] == 'module':
            name = k[7:]  # remove `module.`
        else:
            name = k
        if k == 'RCNN_bbox_pred.bias' or k == 'RCNN_bbox_pred.weight':
           continue
        new_state_dict[name] = v
    origin_state_dict.update(new_state_dict)
    fasterRCNN.load_state_dict(origin_state_dict)


    fasterRCNN.cuda()
    fasterRCNN = nn.DataParallel(fasterRCNN)


    meter_names = ['batch_time',  'loss','rpn_cls_loss','rpn_box_loss','rcnn_cls_loss','rcnn_box_loss','acc','pos_recall','neg_recall','neg_num','pos_num','rpn_acc','rpn_pos_recall','rpn_neg_recall','rpn_neg_num','rpn_pos_num']
    meters = {name: AverageMeter() for name in meter_names}
    fasterRCNN.train()
    for epoch in range(config.TRAIN.begin_epoch,config.TRAIN.end_epoch):
        for i, (data, valid_range, im_info,gt_boxes) in enumerate(train_loader):

            t0 = time.time()
            im_data.data.resize_(data.size()).copy_(data).float()
            valid_range.data.resize_(valid_range.size()).copy_(valid_range).float()
            im_info.data.resize_(im_info.size()).copy_(im_info).float()
            gt_boxes.data.resize_(gt_boxes.size()).copy_(gt_boxes).float()

            fasterRCNN.zero_grad()
            rois, rpn_cls_prob, rpn_label, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = fasterRCNN(im_data, im_info,valid_range, gt_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            pos_recall, neg_recall, acc, pos_num, neg_num = pos_neg_recall(cls_prob, rois_label)

            rpn_pos_recall, rpn_neg_recall, rpn_acc, rpn_pos_num, rpn_neg_num = pos_neg_recall(rpn_cls_prob, rpn_label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            meters['loss'].update(loss.item(), 1)
            meters['batch_time'].update(t1 - t0, 1)
            meters['rpn_cls_loss'].update(rpn_loss_cls.data.mean(), 1)
            meters['rpn_box_loss'].update(rpn_loss_box.data.mean(), 1)
            meters['rcnn_cls_loss'].update(RCNN_loss_cls.data.mean(), 1)
            meters['rcnn_box_loss'].update(RCNN_loss_bbox.data.mean(), 1)
            meters['acc'].update(acc, 1)
            meters['neg_recall'].update(neg_recall, 1)
            meters['pos_recall'].update(pos_recall, 1)
            meters['pos_num'].update(pos_num, 1)
            meters['neg_num'].update(neg_num, 1)
            meters['rpn_acc'].update(rpn_acc, 1)
            meters['rpn_neg_recall'].update(rpn_neg_recall, 1)
            meters['rpn_pos_recall'].update(rpn_pos_recall, 1)
            meters['rpn_pos_num'].update(rpn_pos_num, 1)
            meters['rpn_neg_num'].update(rpn_neg_num, 1)

            if i % 100 == 0:

                print(epoch, i,
                      'Batch_time:  %.4f  lr: %.4f sum_Loss: %.4f rpn_cls_loss: %.4f rpn_box_loss: %.4f  rcnn_cls_loss: %.4f  rcnn_box_loss: %.4f  pos_recall: %.4f   neg_recall: %.4f  acc: %.4f pos_num: %5d neg_num: %5d rpn_pos_recall: %.4f   rpn_neg_recall: %.4f  rpn_acc: %.4f rpn_pos_num: %5d rpn_neg_num: %5d' % (
                      meters['batch_time'].avg, lr,meters['loss'].avg, meters['rpn_cls_loss'].avg,
                      meters['rpn_box_loss'].avg, meters['rcnn_cls_loss'].avg, meters['rcnn_box_loss'].avg,
                      meters['pos_recall'].avg, meters['neg_recall'].avg, meters['acc'].avg, meters['pos_num'].avg,
                      meters['neg_num'].avg, meters['rpn_pos_recall'].avg, meters['rpn_neg_recall'].avg,
                      meters['rpn_acc'].avg, meters['rpn_pos_num'].avg, meters['rpn_neg_num'].avg))
                for k in meters.keys():
                    meters[k].reset()


            if i!=0 and i % 1000 == 0:
                save_name = os.path.join('output','nofix_{}_{}.pth'.format(epoch, i))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': fasterRCNN.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, save_name)

        #if epoch % 1 == 0:
        adjust_learning_rate(optimizer, 0.1)
        lr *= 0.1

        save_name = os.path.join('output', 'nofix_{}.pth'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name)










