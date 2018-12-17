import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
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


def train_one_batch(train_model,optimizer,meters,data, valid_range, im_info,label, bbox_target, bbox_weight, gt_boxes,epoch_index,batch_index):

    t0 = time.time()
    data_var = data.float().cuda()
    valid_range_var = valid_range.float().cuda()
    im_info_var = im_info.float().cuda()
    label_var = label.float().cuda()
    bbox_target_var = bbox_target.float().cuda()
    bbox_weight_var = bbox_weight.float().cuda()
    gt_boxes_var = gt_boxes.float().cuda()

    rois, rpn_cls_prob,rpn_label,cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label = train_model(data_var, im_info_var, valid_range_var, label_var, bbox_target_var, bbox_weight_var, gt_boxes_var)

    pos_recall, neg_recall, acc,pos_num,neg_num = pos_neg_recall(cls_prob, rois_label)
    rpn_pos_recall, rpn_neg_recall, rpn_acc, rpn_pos_num, rpn_neg_num = pos_neg_recall(rpn_cls_prob, rpn_label)

    #print(rpn_loss_cls.mean(),rpn_loss_box.mean() ,RCNN_loss_cls.mean() ,RCNN_loss_bbox.mean())
    optimizer.zero_grad()
    loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
    loss.backward()
    optimizer.step()
    t1 = time.time()

    meters['loss'].update(loss.item(),1)
    meters['batch_time'].update(t1-t0,1)
    meters['rpn_cls_loss'].update(rpn_loss_cls.data.mean(),1)
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

    if batch_index%100==0:
        print(epoch_index,batch_index,'Batch_time:  %.4f  sum_Loss: %.4f rpn_cls_loss: %.4f rpn_box_loss: %.4f  rcnn_cls_loss: %.4f  rcnn_box_loss: %.4f  pos_recall: %.4f   neg_recall: %.4f  acc: %.4f pos_num: %5d neg_num: %5d rpn_pos_recall: %.4f   rpn_neg_recall: %.4f  rpn_acc: %.4f rpn_pos_num: %5d rpn_neg_num: %5d'   %(meters['batch_time'].avg, meters['loss'].avg,meters['rpn_cls_loss'].avg,meters['rpn_box_loss'].avg,meters['rcnn_cls_loss'].avg,meters['rcnn_box_loss'].avg,meters['pos_recall'].avg,meters['neg_recall'].avg,meters['acc'].avg,meters['pos_num'].avg,meters['neg_num'].avg,meters['rpn_pos_recall'].avg,meters['rpn_neg_recall'].avg,meters['rpn_acc'].avg,meters['rpn_pos_num'].avg,meters['rpn_neg_num'].avg) )
        for k in meters.keys():
            meters[k].reset()

    #print(epoch_index,batch_index,rpn_loss_cls.mean().data[0], rpn_loss_box.mean().data[0] , RCNN_loss_cls.mean().data[0] , RCNN_loss_bbox.mean().data[0])

