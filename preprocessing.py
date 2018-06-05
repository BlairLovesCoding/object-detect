from __future__ import division
import os, sys, time
import _pickle
import itertools
sys.path.append('/Users/Blair/cocoapi/PythonAPI')

# matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('jpg')
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from math import ceil, floor


class DataLoader:
    def __init__(self, dataDir, dataType, category):
        self.dataDir = dataDir
        self.dataType = dataType
        self.cat = category
        # self.feats = []
        # self.label = []
        self.pos = []
        self.nega = []

    def get_bboxes(self, img, ss, num_rects=2000):
        try:
            ss.setBaseImage(img)
            # ss.switchToSelectiveSearchQuality() # good quality search
            ss.switchToSelectiveSearchFast()  # fast search
            # t1 = time.time()
            rects = ss.process()
            # t1 = time.time() - t1
            return rects[:num_rects]
        except KeyboardInterrupt:
            print('keyboard interrupt')
            sys.exit()
        except:
            return None

    # IoU
    def iou(self, rect1, rect2):  # rect = [x, y, w, h]
        x1, y1, w1, h1 = rect1
        X1, Y1 = x1 + w1, y1 + h1
        x2, y2, w2, h2 = rect2
        X2, Y2 = x2 + w2, y2 + h2
        a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
        a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
        x_int = max(x1, x2)
        X_int = min(X1, X2)
        y_int = max(y1, y2)
        Y_int = min(Y1, Y2)
        a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0
        if x_int > X_int or y_int > Y_int:
            a_int = 0.0
        return a_int / (a1 + a2 - a_int)

    # nearest neighbor in 1-based indexing
    def _nnb_1(self, x):
        x1 = int(floor((x + 8) / 16.0))
        x1 = max(1, min(x1, 13))
        return x1

    def project_onto_feature_space(self, rect, image_dims):
        # project bounding box onto conv net
        # @param rect: (x, y, w, h)
        # @param image_dims: (imgx, imgy), the size of the image
        # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

        # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
        imgx, imgy = image_dims
        x, y, w, h = rect
        # scale to 224 x 224, standard input size.
        x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
        x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
        px = self._nnb_1(x + 1) - 1  # inclusive
        py = self._nnb_1(y + 1) - 1  # inclusive
        px1 = self._nnb_1(x1 + 1)  # exclusive
        py1 = self._nnb_1(y1 + 1)  # exclusive

        return [px, py, px1, py1]

    class Featurizer:
        dim = 11776  # for small features

        def __init__(self):
            # pyramidal pooling of sizes 1, 3, 6
            self.pool1 = nn.AdaptiveMaxPool2d(1)
            self.pool3 = nn.AdaptiveMaxPool2d(3)
            self.pool6 = nn.AdaptiveMaxPool2d(6)
            self.lst = [self.pool1, self.pool3, self.pool6]

        def featurize(self, projected_bbox, image_features):
            # projected_bbox: bbox projected onto final layer
            # image_features: C x W x H tensor : output of conv net
            full_image_features = torch.from_numpy(image_features)
            x, y, x1, y1 = projected_bbox
            crop = full_image_features[:, x:x1, y:y1]
            #         return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
            #                           self.pool6(crop).view(-1)], dim=0) # returns torch Variable
            return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
                              self.pool6(crop).view(-1)], dim=0).data.numpy()  # returns numpy array

    def load(self, num_img, ratio_nega):
        t1 = time.time()
        annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)  # annotations

        # initialize COCO api for instance annotations. This step takes several seconds each time.
        coco = COCO(annFile)

        cats = coco.loadCats(coco.getCatIds())  # categories
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}  # category id to name mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in cats}  # category name to id mapping

        cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
        cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

        true_id = cat_name_to_id[self.cat]

        # read features:
        [img_ids, feats] = _pickle.load(open(os.path.join(self.dataDir, 'features2_small', '{}.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')
        [img_list, bboxes] = _pickle.load(open(os.path.join(self.dataDir, 'bboxes2', '{}_bboxes.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')
        # print(self.dataType, ":", img_ids == img_list)
        n = len(img_ids)
        count = 0
        # i = 0

        for i in range(n):
            # if bbox is None or len(bbox) == 0:
            #     print("Discard this img for consideration.")
            #     i += 1
            #     continue
            if count == num_img:
                break

            img_id = img_ids[i]
            img = coco.loadImgs([img_id])[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            categories = set([ann['category_id'] for ann in anns])
            if categories.__contains__(true_id) is False:
                continue

            true_bboxes = []
            for ann in anns:
                if ann['category_id'] == true_id:
                    true_bboxes += [ann['bbox']]
            #
            # cv2.setNumThreads(8)
            # ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            #
            # # run selective search on the image
            # img_cv = cv2.imread('%s/%s/%s' % (self.dataDir, self.dataType, img['file_name']))
            # num_rects = 2000
            # bboxes = self.get_bboxes(img_cv, ss, num_rects=num_rects)
            if bboxes[i] is None:  ### OpenCV has thrown an error. Discard image.
                print('Discard image from consideration.')
                continue

            count += 1
            img_feats = feats[i]
            img_pil = Image.open(
                '%s/%s_2/%s' % (self.dataDir, self.dataType, img['file_name']))  # make sure data dir is correct
            # i += 1
            prepos = []
            nega = []
            isNega = True

            for r in bboxes[i]:
                for true_rect in true_bboxes:
                    iou = self.iou(true_rect, r)
                    if iou > 0.5:
                        prepos.append(r)
                        isNega = False
                        break
                if isNega:
                    nega.append(r)
                isNega = True

            # print("# bbox: ", len(bboxes))
            # print("first # pos: ", len(prepos))
            # print("first # nega: ", len(nega))

            if len(prepos) == 0:
                count -= 1
                continue

            num_pos = min(10 * len(true_bboxes), len(prepos))
            prepos = random.sample(prepos, num_pos)
            for item in prepos:
                featurizer = self.Featurizer()
                projected_bbox = self.project_onto_feature_space(item, img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                self.pos.append(bbox_feats.flatten())

            prepos.clear()
            # print("after # pos: ", num_pos)
            num_nega = min(ratio_nega * num_pos, len(nega))
            nega = random.sample(nega, num_nega)
            # print("after # nega: ", len(nega))

            for item in nega:
                featurizer = self.Featurizer()
                projected_bbox = self.project_onto_feature_space(item, img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                self.nega.append(bbox_feats.flatten())

            nega.clear()

            # print("%d seconds, one img done" % (end - start))

        print("%d img processed" % count)
        self.pos = np.array(self.pos)
        self.nega = np.array(self.nega)
        # print(self.dataType, ":", self.pos.shape)
        # print(self.dataType, ":", self.nega.shape)
        t2 = time.time()
        print("%d seconds, data pre-processing done" % (t2 - t1))

    def load_nega(self, num_img, num_nega):
        t1 = time.time()
        annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)  # annotations

        # initialize COCO api for instance annotations. This step takes several seconds each time.
        coco = COCO(annFile)

        cats = coco.loadCats(coco.getCatIds())  # categories
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}  # category id to name mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in cats}  # category name to id mapping

        cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
        cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

        true_id = cat_name_to_id[self.cat]

        # read features:
        [img_ids, feats] = _pickle.load(open(os.path.join(self.dataDir, 'features2_small', '{}.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')
        [img_list, bboxes] = _pickle.load(open(os.path.join(self.dataDir, 'bboxes2', '{}_bboxes.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')

        n = len(img_ids)
        count = 0
        # i = 0
        res = []
        per_img = ((num_nega // num_img) + 1) * 100

        for i in range(n):
            if bboxes[i] is None or len(bboxes[i]) == 0:
                print("Discard this img for consideration.")
                # i += 1
                continue
            if count == num_img:
                break

            img_id = img_ids[i]
            img = coco.loadImgs([img_id])[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            categories = set([ann['category_id'] for ann in anns])
            if categories.__contains__(true_id) is False:
                continue

            true_bboxes = []
            for ann in anns:
                if ann['category_id'] == true_id:
                    true_bboxes += [ann['bbox']]

            count += 1
            img_feats = feats[i]
            img_pil = Image.open(
                '%s/%s_2/%s' % (self.dataDir, self.dataType, img['file_name']))  # make sure data dir is correct
            # i += 1
            nega = []
            isNega = True
            for r in bboxes[i]:
                for true_rect in true_bboxes:
                    iou = self.iou(true_rect, r)
                    if iou > 0.5:
                        isNega = False
                        break
                if isNega:
                    nega.append(r)
                isNega = True

            nega = random.sample(nega, per_img)

            for item in nega:
                featurizer = self.Featurizer()
                projected_bbox = self.project_onto_feature_space(item, img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                res.append(bbox_feats.flatten())
        # print("res: ", len(res))
        res = random.sample(res, num_nega)
        # print("res: ", len(res))
        t2 = time.time()
        print("%d seconds, hard negative mining done" % (t2 - t1))
        return np.array(res)

    def load_test(self, num_img):
        t1 = time.time()
        annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)  # annotations

        # initialize COCO api for instance annotations. This step takes several seconds each time.
        coco = COCO(annFile)

        cats = coco.loadCats(coco.getCatIds())  # categories
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}  # category id to name mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in cats}  # category name to id mapping

        cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
        cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

        true_id = cat_name_to_id[self.cat]

        # read features:
        [img_ids, feats] = _pickle.load(open(os.path.join(self.dataDir, 'features2_small', '{}.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')
        [img_list, bboxes] = _pickle.load(open(os.path.join(self.dataDir, 'bboxes_retrieval', '{}_bboxes_retrieval.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')

        n = len(img_ids)
        count = 0
        i = 0
        feat = []
        label = []
        # per_img = (num_test // num_img) + 1
        foundPos = False

        for bbox in bboxes:

            if bbox is None or len(bbox) == 0:
                print("Discard this img for consideration.")
                i += 1
                continue
            if count == num_img:
                break

            count += 1
            img_id = img_ids[i]
            img = coco.loadImgs([img_id])[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            categories = set([ann['category_id'] for ann in anns])
            if categories.__contains__(true_id) is False:
                # i += 1
                # continue
                true_bboxes = []
            else:
                true_bboxes = []
                for ann in anns:
                    if ann['category_id'] == true_id:
                        true_bboxes += [ann['bbox']]

            img_feats = feats[i]
            img_pil = Image.open(
                '%s/%s_2/%s' % (self.dataDir, self.dataType, img['file_name']))  # make sure data dir is correct
            i += 1

            # if foundPos:
            #     bbox = random.sample(list(bbox), 10)

            for item in bbox:
                featurizer = self.Featurizer()
                projected_bbox = self.project_onto_feature_space(item, img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                feat.append(bbox_feats.flatten())
                isNega = True
                if len(true_bboxes) == 0:
                    label.append(0)
                else:
                    for true_rect in true_bboxes:
                        if self.iou(item, true_rect) > 0.5:
                            label.append(1)
                            foundPos = True
                            isNega = False
                            break
                    if isNega:
                        label.append(0)

        # feat = feat[:num_test]
        # label = label[:num_test]
        # print(len(feat))
        # print(len(label))
        print(foundPos)
        t2 = time.time()
        print("%d seconds, test loading done" % (t2 - t1))
        return np.array(feat), np.array(label).reshape([len(label), 1])

