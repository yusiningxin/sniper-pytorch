from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from bbox.bbox_transform import *
import numpy.random as npr
#from ..utils.config import cfg
#from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch



class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, cfg):
        super(_ProposalTargetLayer, self).__init__()

        self.FG_THRESH = cfg.TRAIN.FG_THRESH
        self.BG_THRESH_LO = cfg.TRAIN.BG_THRESH_LO
        self.BG_THRESH_HI = cfg.TRAIN.BG_THRESH_HI

        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_MEANS).cuda()
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_STDS).cuda()

        # self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, valid_range):

        #self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        #self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        #self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        #print(all_rois.shape,gt_boxes.shape,self.BBOX_NORMALIZE_MEANS.shape)
        batch_size = all_rois.size(0)
        #print("batch size",batch_size)

        all_rois = self.append_gtbox_to_roi(gt_boxes,all_rois,valid_range,batch_size)

        rois_per_image = all_rois.size(1)
        gt_boxes = gt_boxes.float()
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(all_rois, gt_boxes,rois_per_image)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    def append_gtbox_to_roi(self,gt_boxes,all_rois,valid_range,batch_size):

        widths = gt_boxes[:, :, 2] - gt_boxes[:, :, 0]
        heights = gt_boxes[:, :, 3] - gt_boxes[:, :, 1]
        area = (widths * heights).cpu().numpy()
        gt_boxes_np = gt_boxes.cpu().numpy()

        for i in range(batch_size):
            min_area = valid_range[i][0] * valid_range[i][0]
            max_area = valid_range[i][1] * valid_range[i][1]
            idx = np.where((area[i] >= min_area) & (area[i] <= max_area))[0]
            gt_idx = np.where(gt_boxes_np[i,:,4]!=-1)[0]
            idx = np.intersect1d(idx, gt_idx)
            if len(idx)>0:
                idx = torch.from_numpy(idx).long().cuda()
                # print(len(idx),min_area,max_area,area[i],torch.index_select(gt_boxes[i],0,idx).shape)
                # print(all_rois[i,-len(idx):].shape)
                #all_rois[i,-len(idx):] = torch.index_select(gt_boxes[i],0,idx)
                all_rois[i, -len(idx):] = torch.index_select(gt_boxes[i], 0, idx)
        return all_rois






    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)
        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).
        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = 1.0

        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        #     # Optionally normalize targets by a precomputed mean and stdev

        targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))/ self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, rois_per_image):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        #print(max_overlaps.shape,gt_assignment.shape,max_overlaps,gt_assignment)
        batch_size = overlaps.size(0)


        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        #print(offset,offset.shape)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:, :, 4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)
        #print(labels)
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= self.FG_THRESH).view(-1)
            bg_inds = torch.nonzero((max_overlaps[i] < self.BG_THRESH_HI) & (max_overlaps[i] >= self.BG_THRESH_LO)).view(-1)

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            #print(len(fg_inds),len(bg_inds))
            labels_batch[i][len(fg_inds):] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i, :, 0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])
        bbox_targets, bbox_inside_weights = self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights