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
from data_utils.generate_anchor import generate_anchors
from bbox.bbox_transform import bbox_transform_inv
from bbox.bbox_transform import clip_boxes_batch
from nms.nms import *
# from model.utils.config import cfg
# from .generate_anchors import generate_anchors
# from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms


DEBUG = False


class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self,cfg):
        super(_ProposalLayer, self).__init__()
        self.scales = np.array(cfg.network.ANCHOR_SCALES, dtype=np.float32)
        self.ratios = cfg.network.ANCHOR_RATIOS
        self._feat_stride = cfg.network.RPN_FEAT_STRIDE

        self._anchors = torch.from_numpy(generate_anchors(base_size=self._feat_stride, ratios=list(self.ratios),scales=list(self.scales)))

        self._num_anchors = cfg.network.NUM_ANCHORS
        self.pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N
        self.post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N
        self.nms_thresh = cfg.TRAIN.RPN_NMS_THRESH
        self.min_size = cfg.TRAIN.RPN_MIN_SIZE

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        scores = input[0][:, self._num_anchors:, :, :]

        bbox_deltas = input[1]
        im_info = input[2]
        valid_range = input[3]

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)

        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)


        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes_batch(proposals, im_info[:,:2], batch_size)

        self.filter_proposal_by_range(scores,proposals,valid_range,batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, self.post_nms_topN, 5).zero_()

        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if self.pre_nms_topN > 0 and self.pre_nms_topN < scores_keep.numel():
                order_single = order_single[:self.pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = gpu_nms(torch.cat((proposals_single, scores_single), 1).cpu().numpy(), self.nms_thresh)
            #print('nms',len(keep_idx_i))
            keep_idx_i = torch.from_numpy(np.array(keep_idx_i)).cuda()
            keep_idx_i = keep_idx_i.long().view(-1)

            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]

            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single

        return output

    def filter_proposal_by_range(self,scores, proposals, valid_range,batch_size):
        widths = proposals[:,:, 2] - proposals[:,:, 0]
        heights = proposals[:,:, 3] - proposals[:,:, 1]
        area = (widths * heights).cpu().numpy()
        for i in range(batch_size):
            min_area = valid_range[i][0]*valid_range[i][0]
            max_area = valid_range[i][1]*valid_range[i][1]
            idx = np.where((area[i] < min_area) | (area[i]>max_area))[0]
            scores[i][idx] = -1



    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
        return keep