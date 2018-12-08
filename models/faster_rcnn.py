import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


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

        self.bn1 = nn.BatchNorm2d(num_features=init_feature, affine=False, eps=2e-5)
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
    def __init__(self,cfg,units = (3,4,23,4),filter_list = [64, 256, 512, 1024, 2048],fix_bn = False,momentum= 0.95,is_train = True):
        super(FasterRCNN, self).__init__()

        self.num_anchors = cfg.network.NUM_ANCHORS
        # shared conv layers
        self.conv_feat = resnetc4(3,units,filter_list,fix_bn,momentum,fp16=cfg.TRAIN.fp16)

        # res5
        self.relut = resnetc5(units = (3,4,23,4),filter_list = [64, 256, 512, 1024, 2048],num_stage=4,momentum=momentum,deform=False)

        #get_rpn
        self.rpn_conv = nn.Conv2d(filter_list[4]+filter_list[3], 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpn_relu = nn.ReLU(inplace=True)
        self.rpn_cls_score = nn.Conv2d(512, 2 * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=False)
        self.rpn_bbox_pred = nn.Conv2d(512, 4 * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, data):
        conv_feat= self.conv_feat(data)
        relut = self.relut(conv_feat)

        relu1 = torch.cat([conv_feat,relut],1)
        rpn_mid = self.rpn_relu(self.rpn_conv(relu1))

        rpn_cls_score = self.rpn_cls_score(rpn_mid)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_mid)

        input_shape = rpn_cls_score.shape

        rpn_cls_score = rpn_cls_score.view(input_shape[0],2,self.num_anchors,-1)
        rpn_cls_prob = F.softmax(rpn_cls_score)



        print(rpn_cls_prob.shape,rpn_bbox_pred.shape)


    # def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):
    #     rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
    #     rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
    #     x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
    #     x = network.np_to_variable(x, is_cuda=True)
    #     return x.view(-1, 5)
    #
    # @staticmethod
    # def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
    #     """
    #     rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
    #     gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    #     gt_ishard: (G, 1), 1 or 0 indicates difficult or not
    #     dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
    #     im_info: a list of [image_height, image_width, scale_ratios]
    #     _feat_stride: the downsampling ratio of feature map to the original input image
    #     anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    #     ----------
    #     Returns
    #     ----------
    #     rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    #     rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
    #                     that are the regression objectives
    #     rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
    #     rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
    #     beacuse the numbers of bgs and fgs mays significiantly different
    #     """
    #     rpn_cls_score = rpn_cls_score.data.cpu().numpy()
    #     rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
    #         anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)
    #
    #     rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
    #     rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
    #     rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
    #     rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)
    #
    #     return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


