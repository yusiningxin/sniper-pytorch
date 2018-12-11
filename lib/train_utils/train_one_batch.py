import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


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

def train_one_batch(train_model,optimizer,data, valid_range, im_info,label, bbox_target, bbox_weight, gt_boxes,epoch_index,batch_index):
    data_var = torch.autograd.Variable(data).cuda()
    valid_range_var = torch.autograd.Variable(valid_range).cuda()
    im_info_var = torch.autograd.Variable(im_info).cuda()
    label_var = torch.autograd.Variable(label).cuda()
    bbox_target_var = torch.autograd.Variable(bbox_target).cuda()
    bbox_weight_var = torch.autograd.Variable(bbox_weight).cuda()
    gt_boxes_var = torch.autograd.Variable(gt_boxes).cuda()

    rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_bbox, rois_label = train_model(data_var,im_info_var,valid_range_var,label_var,bbox_target_var,bbox_weight_var,gt_boxes_var)

    # optimizer.zero_grad()
    # RCNN_loss_bbox.backward(retain_graph=True)
    # RCNN_loss_cls.backward()
    # optimizer.step()
    print(RCNN_loss_cls.data, RCNN_loss_bbox.data)

