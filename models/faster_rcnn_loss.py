import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from proposal_layer_py import _ProposalLayer
from proposal_target_layer import _ProposalTargetLayer
from roi_pooling.modules.roi_pool import _RoIPooling


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    #print(type(bbox_outside_weights),type(in_loss_box),bbox_outside_weights.shape,in_loss_box.shape)

    out_loss_box = bbox_outside_weights * in_loss_box

    loss_box = out_loss_box
    print(loss_box.shape,loss_box[loss_box!=0])
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
        #print(i,loss_box.shape)
    print(loss_box.shape)
    loss_box = loss_box.mean()
    print(loss_box)
    return loss_box


class residual_unit(nn.Module):
    def __init__(self,init_feature, num_filter, stride, dim_match, momentum, fix_bn):
        super(residual_unit, self).__init__()

        self.dim_match = dim_match

        if fix_bn:
            self.bn1 = nn.BatchNorm2d(num_features = init_feature,affine = True,eps = 2e-5)
        else:
            self.bn1 = nn.BatchNorm2d(num_features = init_feature,affine = True,eps = 2e-5,momentum = momentum)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(init_feature, int(num_filter*0.25), kernel_size=1, stride=1,padding=0, bias=False)


        if fix_bn:
            self.bn2 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5)
        else:
            self.bn2 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5,momentum = momentum)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(num_filter*0.25), int(num_filter*0.25), kernel_size=3, stride=stride,padding=1, bias=False)

        if fix_bn:
            self.bn3 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5)
        else:
            self.bn3 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5,momentum = momentum)

        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(num_filter*0.25), num_filter, kernel_size=1, stride=1,padding=0, bias=False)
        if not dim_match:
            self.conv_short = nn.Conv2d(init_feature, num_filter, kernel_size=1, stride=stride,padding=0, bias=False)

    def forward(self,data):
        out = self.relu1(self.bn1(data))
        short = out
        out = self.relu2(self.bn2(self.conv1(out)))
        out = self.relu3(self.bn3(self.conv2(out)))
        out = self.conv3(out)

        if self.dim_match:
            #print("match residual unit: ", out.shape, short.shape)
            return torch.add(data , out)
        else:
            short = self.conv_short(short)
            #print("not residual unit: ", out.shape, short.shape)
            return torch.add(short,out)


class residual_unit_dilate(nn.Module):
    def __init__(self,init_feature, num_filter, stride, dim_match, momentum, fix_bn=False):
        super(residual_unit_dilate, self).__init__()

        self.dim_match = dim_match

        if fix_bn:
            self.bn1 = nn.BatchNorm2d(num_features = init_feature,affine = True,eps = 2e-5)
        else:
            self.bn1 = nn.BatchNorm2d(num_features = init_feature,affine = True,eps = 2e-5,momentum = momentum)

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(init_feature, int(num_filter*0.25), kernel_size=1, stride=1,padding=0, bias=False)


        if fix_bn:
            self.bn2 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5)
        else:
            self.bn2 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5,momentum = momentum)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(num_filter*0.25), int(num_filter*0.25), kernel_size=3, stride=stride, dilation=2,padding=2, bias=False)

        if fix_bn:
            self.bn3 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5)
        else:
            self.bn3 = nn.BatchNorm2d(num_features = int(num_filter*0.25),affine = True,eps = 2e-5,momentum = momentum)

        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(num_filter*0.25), num_filter, kernel_size=1, stride=1,padding=0, bias=False)
        if not dim_match:
            self.conv_short = nn.Conv2d(init_feature, num_filter, kernel_size=1, stride=stride,padding=0, bias=False)

    def forward(self,data):
        out = self.relu1(self.bn1(data))
        short = out
        out = self.relu2(self.bn2(self.conv1(out)))
        out = self.relu3(self.bn3(self.conv2(out)))
        out = self.conv3(out)

        if self.dim_match:
            #print("match residual unit: ", out.shape, short.shape)
            return torch.add(data , out)
        else:
            short = self.conv_short(short)
            #print("not residual unit: ", out.shape, short.shape)
            return torch.add(short,out)



class resnetc4(nn.Module):
    def __init__(self,init_feature,units,filter_list,fix_bn,momentum,fp16):
        super(resnetc4, self).__init__()
        num_stage = len(units)

        self.bn1 = nn.BatchNorm2d(num_features=init_feature, affine=True, eps=2e-5)
        self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=7, stride=2, padding=3, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=filter_list[0], affine=True, eps=2e-5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride=2, padding=1)

        current_feature = filter_list[0]
        for i in range(num_stage - 1):
            if i==0:
                self.blocks_layer = nn.Sequential(OrderedDict([('stage%d_unit%d' %(i+1,1),residual_unit(current_feature,filter_list[i+1],1 ,False,momentum,fix_bn=True))]))
            else:
                self.blocks_layer.add_module('stage%d_unit%d' %(i+1,1),residual_unit(current_feature,filter_list[i+1],2,False,momentum,fix_bn=False))

            current_feature = filter_list[i+1]
            for j in range(units[i]-1):
                self.blocks_layer.add_module('stage%d_unit%d' %(i+1,j+2),residual_unit(current_feature,filter_list[i+1],1,True,momentum,fix_bn=(i == 0)))


    def forward(self,data):
        out = self.bn1(data)
        out = self.relu(self.bn2(self.conv1(out)))
        out = self.maxpool(out)
        out = self.blocks_layer(out)
        return out


class resnetc5(nn.Module):
    def __init__(self,units,filter_list,num_stage,momentum,deform=False):
        super(resnetc5, self).__init__()

        self.block_layer = nn.Sequential(OrderedDict([('stage%d_unit%d'% (num_stage,1),residual_unit_dilate(filter_list[num_stage-1],filter_list[num_stage], 1, False,momentum))]))

        for i in range(units[num_stage-1] - 1):
            if not deform:
                self.block_layer.add_module('stage%d_unit%d' % (num_stage, i + 2),residual_unit_dilate(filter_list[num_stage],filter_list[num_stage], 1, True,momentum))

    def forward(self,data):
        out = self.block_layer(data)
        return out




class FasterRCNN(nn.Module):
    def __init__(self,cfg,units = (3,4,23,3),filter_list = [64, 256, 512, 1024, 2048],fix_bn = False,momentum= 0.95,is_train=True):
        super(FasterRCNN, self).__init__()
        self.is_train = is_train
        self.cfg = cfg
        self.num_anchors = cfg.network.NUM_ANCHORS
        # shared conv layers
        self.conv_feat = resnetc4(3,units,filter_list,fix_bn,momentum,fp16=cfg.TRAIN.fp16)

        self.BBOX_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_STDS)

        # res5
        self.relut = resnetc5(units = (3,4,23,3),filter_list = [64, 256, 512, 1024, 2048],num_stage=4,momentum=momentum,deform=False)

        # conv
        self.conv_new = nn.Conv2d(filter_list[4]+filter_list[3], 256 , kernel_size=1, stride=1, padding=0, bias=False)
        # relu
        self.relu_new = nn.ReLU(inplace=True)

        #get_rpn
        self.rpn_conv = nn.Conv2d(filter_list[4]+filter_list[3], 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpn_relu = nn.ReLU(inplace=True)
        self.rpn_cls_score = nn.Conv2d(512, 2 * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=False)
        self.rpn_bbox_pred = nn.Conv2d(512, 4 * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=False)


        # proposal layer for rpn
        self.RPN_proposal = _ProposalLayer(cfg)

        # proposal target generate
        self.RCNN_proposal_target = _ProposalTargetLayer(cfg)

        # roi pooling
        self.RCNN_roi_pool = _RoIPooling(7,7, 1.0 / 16.0)

        # fc
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        #result
        self.cls_score = nn.Linear(1024, self.cfg.dataset.NUM_CLASSES)
        self.bbox_pred = nn.Linear(1024, 4)


    def forward(self, data,im_info,valid_range=None,label = None,bbox_target = None,bbox_weight = None,gt_boxes = None,im_ids = None):

        batch_size = data.size(0)

        conv_feat= self.conv_feat(data)
        relut = self.relut(conv_feat)
        relu1 = torch.cat([conv_feat,relut],1)
        conv_feat_new = self.relu_new(self.conv_new(relu1))

        rpn_mid = self.rpn_relu(self.rpn_conv(relu1))

        rpn_cls_score = self.rpn_cls_score(rpn_mid)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_mid)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, 2*self.num_anchors)


        #print(rpn_cls_prob.shape, rpn_bbox_pred.shape,label.shape,bbox_target.shape,bbox_weight.shape)
        # proposal layer
        rois = self.RPN_proposal((rpn_cls_prob, rpn_bbox_pred, im_info, valid_range))

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        if self.is_train:
            roi_data = self.RCNN_proposal_target(rois,gt_boxes,valid_range)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois = rois.view(-1,5)
            rois_label = rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2))
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))


            pooled_feat = self.RCNN_roi_pool(conv_feat_new, rois)
            pooled_feat = pooled_feat.view(pooled_feat.size(0),-1)

            out = self.fc1(pooled_feat)
            out = self.fc2(out)
            cls_score = self.cls_score(out)
            bbox_pred = self.bbox_pred(out)
            cls_prob = F.softmax(cls_score,1)

            # RCNN loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # rois_outside_ws = torch.ones(rois_inside_ws.shape).float().cuda()
            # rois_outside_ws = rois_outside_ws * batch_size * 1.0 / (len(rpn_keep))

            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


            # RPN cls Loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = label.view(batch_size, -1)

            rpn_keep = rpn_label.view(-1).ne(-1).nonzero().view(-1)

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)

            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep)
            rpn_label = rpn_label.long()

            rpn_cls_prob = F.softmax(rpn_cls_score,1)
            rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            # RPN bbox loss
            bbox_outside_weight = torch.ones(bbox_weight.shape).float().cuda()
            bbox_outside_weight = bbox_outside_weight*batch_size*1.0/(len(rpn_keep))

            rpn_loss_bbox = _smooth_l1_loss(rpn_bbox_pred, bbox_target.float(), bbox_weight.float(), Variable(bbox_outside_weight), sigma=3, dim=[1,2,3])

            return rois, rpn_cls_prob,rpn_label,cls_prob, bbox_pred, rpn_loss_cls,rpn_loss_bbox,RCNN_loss_cls, RCNN_loss_bbox, rois_label
        else:

            rois = rois.view(-1, 5)

            pooled_feat = self.RCNN_roi_pool(conv_feat_new, rois)
            pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)

            out = self.fc1(pooled_feat)
            out = self.fc2(out)
            cls_score = self.cls_score(out)
            bbox_pred = self.bbox_pred(out)
            cls_prob = F.softmax(cls_score, 1)

            bbox_pred = bbox_pred * self.BBOX_STDS.expand_as(bbox_pred).cuda()

            return rois,cls_prob, bbox_pred,im_ids





    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


