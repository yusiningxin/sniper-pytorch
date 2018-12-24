import torch.utils.data as data
import numpy as np
from multiprocessing import Pool
from data_utils.data_workers import anchor_worker, im_worker, chip_worker
from multiprocessing.pool import ThreadPool
import math
import matplotlib.pyplot as plt

class PytorchIterator(data.Dataset):
    def __init__(self, roidb, config, batch_size=4, threads=8, nGPUs=1, pad_rois_to=400, crop_size=(512, 512),single_size_change=False):
        self.cur_i = 0
        self.roidb = roidb
        self.batch_size = batch_size
        self.pixel_mean = config.network.PIXEL_MEANS
        self.thread_pool = ThreadPool(threads)
        # self.executor_pool = ThreadPoolExecutor(threads)

        self.n_per_gpu = batch_size / nGPUs
        self.batch = None

        self.cfg = config
        self.n_expected_roi = pad_rois_to

        self.pad_label = np.array(-1)
        self.pad_weights = np.zeros((1, 8))
        self.pad_targets = np.zeros((1, 8))
        self.pad_roi = np.array([[0, 0, 100, 100]])
        self.single_size_change = single_size_change


        self.crop_size = crop_size
        self.num_classes = roidb[0]['gt_overlaps'].shape[1]
        self.bbox_means = np.tile(np.array(config.TRAIN.BBOX_MEANS), (self.num_classes, 1))
        self.bbox_stds = np.tile(np.array(config.TRAIN.BBOX_STDS), (self.num_classes, 1))

        if config.TRAIN.WITH_MASK:
            self.label_name.append('gt_masks')

        self.pool = Pool(config.TRAIN.NUM_PROCESS)
        self.epiter = 0
        self.im_worker = im_worker(crop_size=self.crop_size[0], cfg=config)
        self.chip_worker = chip_worker(chip_size=self.crop_size[0], cfg=config)
        self.anchor_worker = anchor_worker(chip_size=self.crop_size[0], cfg=config)
        self.get_chip()
       # self.get_all_data_and_label()

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, item):

        num = item%self.batch_size

        if num == 0:
            self.get_chip_label_per_batch()
        if not self.cfg.TRAIN.ONLY_PROPOSAL:
            return self.data_batch[0][num],self.data_batch[1][num],self.data_batch[2][num],self.label_batch[0][num]
        else:
            return self.data_batch[0][num],self.label_batch[0][num], self.label_batch[1][num], self.label_batch[2][num]



    def get_chip_label_per_batch(self):
        cur_from = self.cur_i
        cur_to = self.cur_i + self.batch_size
        self.cur_i = (self.cur_i + self.batch_size)%len(self.inds)

        roidb = [self.roidb[self.inds[i]] for i in range(cur_from, cur_to)]
        cropids = [self.roidb[self.inds[i]]['chip_order'][self.crop_idx[self.inds[i]] % len(self.roidb[self.inds[i]]['chip_order'])] for i in range(cur_from, cur_to)]
        n_batch = len(roidb)

        ims = []
        for i in range(n_batch):
            ims.append([roidb[i]['image'], roidb[i]['crops'][cropids[i]], roidb[i]['flipped']])

        #print("begin im_work")
        processed_list = self.thread_pool.map_async(self.im_worker.worker, ims)

        for i in range(cur_from, cur_to):
            self.crop_idx[self.inds[i]] = self.crop_idx[self.inds[i]] + 1

        processed_roidb = []
        for i in range(len(roidb)):
            tmp = roidb[i].copy()
            scale = roidb[i]['crops'][cropids[i]][1]
            tmp['im_info'] = [self.crop_size[0], self.crop_size[1], scale]
            processed_roidb.append(tmp)

        worker_data = []
        srange = np.zeros((len(processed_roidb), 2))
        chipinfo = np.zeros((len(processed_roidb), 3))
        for i in range(len(processed_roidb)):
            cropid = cropids[i]
            nids = processed_roidb[i]['props_in_chips'][cropid]
            gtids = np.where(processed_roidb[i]['max_overlaps'] == 1)[0]
            gt_boxes = processed_roidb[i]['boxes'][gtids, :]
            boxes = processed_roidb[i]['boxes'].copy()
            cur_crop = processed_roidb[i]['crops'][cropid][0]
            im_scale = processed_roidb[i]['crops'][cropid][1]
            height = processed_roidb[i]['crops'][cropid][2]
            width = processed_roidb[i]['crops'][cropid][3]
            classes = processed_roidb[i]['max_classes'][gtids]
            if self.cfg.TRAIN.WITH_MASK:
                gt_masks = processed_roidb[i]['gt_masks']

            for scalei, cscale in enumerate(self.cfg.TRAIN.SCALES):
                if scalei == len(self.cfg.TRAIN.SCALES) - 1:
                    # Last or only scale
                    srange[i, 0] = 0 if self.cfg.TRAIN.VALID_RANGES[scalei][0] < 0 else \
                        self.cfg.TRAIN.VALID_RANGES[scalei][0] * im_scale
                    srange[i, 1] = self.crop_size[1] if self.cfg.TRAIN.VALID_RANGES[scalei][1] < 0 else \
                        self.cfg.TRAIN.VALID_RANGES[scalei][1] * im_scale  # max scale
                elif im_scale == cscale:
                    # Intermediate scale
                    srange[i, 0] = 0 if self.cfg.TRAIN.VALID_RANGES[scalei][0] < 0 else \
                        self.cfg.TRAIN.VALID_RANGES[scalei][0] * self.cfg.TRAIN.SCALES[scalei]
                    srange[i, 1] = self.crop_size[1] if self.cfg.TRAIN.VALID_RANGES[scalei][1] < 0 else \
                        self.cfg.TRAIN.VALID_RANGES[scalei][1] * self.cfg.TRAIN.SCALES[scalei]
                    break
            chipinfo[i, 0] = height
            chipinfo[i, 1] = width
            chipinfo[i, 2] = im_scale
            argw = [processed_roidb[i]['im_info'], cur_crop, im_scale, nids, gtids, gt_boxes, boxes,
                    classes.reshape(len(classes), 1)]
            if self.cfg.TRAIN.WITH_MASK:
                argw += [gt_masks]
            worker_data.append(argw)
        #print("begin anchor_work")

        all_labels = self.pool.map(self.anchor_worker.worker, worker_data)

        feat_width = self.crop_size[1] / self.cfg.network.RPN_FEAT_STRIDE
        feat_height = self.crop_size[0] / self.cfg.network.RPN_FEAT_STRIDE

        labels = np.zeros((n_batch, self.cfg.network.NUM_ANCHORS * feat_height * feat_width))
        bbox_targets = np.zeros((n_batch, self.cfg.network.NUM_ANCHORS * 4, feat_height, feat_width))
        bbox_weights = np.zeros((n_batch, self.cfg.network.NUM_ANCHORS * 4, feat_height, feat_width))
        gt_boxes = -np.ones((n_batch, 100, 5))

        if self.cfg.TRAIN.WITH_MASK:
            encoded_masks = -np.ones((n_batch,100,500))

        for i in range(len(all_labels)):
            # labels[i] = all_labels[i][0][0]
            # pids = all_labels[i][2]
            # if len(pids[0]) > 0:
            #     bbox_targets[i][pids[0], pids[1], pids[2]] = all_labels[i][1]
            #     bbox_weights[i][pids[0], pids[1], pids[2]] = 1.0
            gt_boxes[i] = all_labels[i][3]
            # if self.cfg.TRAIN.WITH_MASK:
            #     encoded_masks[i] = all_labels[i][4]

        im_tensor = np.zeros((n_batch, 3, self.crop_size[0], self.crop_size[1]), dtype=np.float32)
        processed_list = processed_list.get()
        for i in range(len(processed_list)):
            im_tensor[i] = processed_list[i]

        self.data_batch = [im_tensor] if self.cfg.TRAIN.ONLY_PROPOSAL else \
            [im_tensor, srange, chipinfo]

        self.label_batch = [labels, bbox_targets, bbox_weights] if self.cfg.TRAIN.ONLY_PROPOSAL else \
            [gt_boxes]

        if self.cfg.TRAIN.WITH_MASK:
            self.label_batch.append(np.array(encoded_masks))
        #self.visualize(im_tensor, gt_boxes)

        # return mx.io.DataBatch(data=self.data, label=self.label, pad=self.getpad(), index=self.getindex(),
        #                        provide_data=self.provide_data, provide_label=self.provide_label)

    def get_chip(self):
        self.cur_i = 0
        self.n_neg_per_im = 2
        self.crop_idx = [0] * len(self.roidb)
        self.chip_worker.reset()

        # Devide the dataset and  extract chips for each part
        n_per_part = int(math.ceil(len(self.roidb) / float(self.cfg.TRAIN.CHIPS_DB_PARTS)))
        chips = []
        # generate chips including all gt_boxes,3 scales
        for i in range(self.cfg.TRAIN.CHIPS_DB_PARTS):
            chips += self.pool.map(self.chip_worker.chip_extractor,
                                   self.roidb[i * n_per_part:min((i + 1) * n_per_part, len(self.roidb))])

        chip_count = 0
        for i, r in enumerate(self.roidb):
            cs = chips[i]
            chip_count += len(cs)
            r['crops'] = cs

        all_props_in_chips = []
        for i in range(self.cfg.TRAIN.CHIPS_DB_PARTS):
            all_props_in_chips += self.pool.map(self.chip_worker.box_assigner,
                                                self.roidb[i * n_per_part:min((i + 1) * n_per_part, len(self.roidb))])

        for ps, cur_roidb in zip(all_props_in_chips, self.roidb):
            cur_roidb['props_in_chips'] = ps[0]
            if self.cfg.TRAIN.USE_NEG_CHIPS:
                cur_roidb['neg_crops'] = ps[1]
                cur_roidb['neg_props_in_chips'] = ps[2]
        chipindex = []
        if self.cfg.TRAIN.USE_NEG_CHIPS:
            # Append negative chips
            for i, r in enumerate(self.roidb):
                cs = r['neg_crops']
                if len(cs) > 0:
                    sel_inds = np.arange(len(cs))
                    if len(cs) > self.n_neg_per_im:
                        sel_inds = np.random.permutation(sel_inds)[0:self.n_neg_per_im]
                    for ind in sel_inds:
                        chip_count = chip_count + 1
                        r['crops'].append(r['neg_crops'][ind])
                        r['props_in_chips'].append(r['neg_props_in_chips'][ind].astype(np.int32))
                for j in range(len(r['crops'])):
                    chipindex.append(i)
        else:
            for i, r in enumerate(self.roidb):
                for j in range(len(r['crops'])):
                    chipindex.append(i)

        print('Total number of extracted chips: {}'.format(chip_count))
        blocksize = self.batch_size
        chipindex = np.array(chipindex)

        if chipindex.shape[0] % blocksize > 0:
            extra = blocksize - (chipindex.shape[0] % blocksize)
            chipindex = np.hstack((chipindex, chipindex[0:extra]))
        print('add extra chips: {}'.format(chipindex.shape))
        allinds = np.random.permutation(chipindex)
        #allinds = chipindex
        self.inds = np.array(allinds, dtype=int)
        for r in self.roidb:
            r['chip_order'] = np.random.permutation(np.arange(len(r['crops'])))
            #r['chip_order'] = range(len(r['crops']))
        print("Get Chip Done")
        #self.epiter = self.epiter + 1
        #self.size = len(self.inds)

    def visualize(self, im_tensor, boxes):
        for imi in range(im_tensor.shape[0]):
            im = np.zeros((im_tensor.shape[2], im_tensor.shape[3], 3), dtype=np.uint8)
            for i in range(3):
                im[:, :, i] = im_tensor[imi, i, :, :] + self.pixel_mean[2 - i]
            # Visualize positives
            plt.imshow(im)
            cboxes = boxes[imi]
            for box in cboxes:
                rect = plt.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1], fill=False,
                                     edgecolor='green', linewidth=3.5)
                plt.gca().add_patch(rect)
            num = np.random.randint(100000)
            plt.savefig('/home/liuqiuyue/debug_0/test_{}_pos.png'.format(num))
            plt.cla()
            plt.clf()
            plt.close()