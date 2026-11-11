"""COCO Boundary AP using the Boundary IoU paper's instance matching rule.

Reference: https://github.com/bowenc0221/boundary-iou-api
For non-crowd GT, matching uses min(mask IoU, boundary IoU); crowd GT keeps
mask IoU. The boundary is the foreground minus its eroded interior, with
zero padding at image edges. Default width: round(0.02 * image diagonal).
This adapter retains pycocotools' AP matching, interpolation and area groups.
"""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def boundary_mask(mask, dilation_ratio=0.02):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not 0 < dilation_ratio <= 1:
        raise ValueError('Expected a 2D binary mask and 0 < dilation_ratio <= 1.')
    width = max(1, round(math.hypot(*mask.shape) * dilation_ratio))
    padded = np.pad(mask, 1, mode='constant')
    interior = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=width)
    return mask - interior[1:-1, 1:-1]


class BoundaryCOCOeval(COCOeval):
    """Standard COCOeval with only the pairwise instance IoU replaced."""

    def __init__(self, coco_gt, coco_dt, dilation_ratio=0.02):
        super().__init__(coco_gt, coco_dt, iouType='segm')
        self.dilation_ratio = dilation_ratio
        self._boundary_cache = {}

    def _boundary_rle(self, annotation, is_ground_truth):
        key = (is_ground_truth, annotation['id'])
        if key not in self._boundary_cache:
            binary = mask_utils.decode(annotation['segmentation'])
            edge = boundary_mask(binary, self.dilation_ratio)
            self._boundary_cache[key] = mask_utils.encode(np.asfortranarray(edge))
        return self._boundary_cache[key]

    def computeIoU(self, imgId, catId):
        region_iou = super().computeIoU(imgId, catId)
        if np.size(region_iou) == 0:
            return region_iou
        if self.params.useCats:
            truth = self._gts[imgId, catId]
            detections = self._dts[imgId, catId]
        else:
            truth = [a for c in self.params.catIds for a in self._gts[imgId, c]]
            detections = [a for c in self.params.catIds for a in self._dts[imgId, c]]
        order = np.argsort([-a['score'] for a in detections], kind='mergesort')
        detections = [detections[i] for i in order[:self.params.maxDets[-1]]]
        crowd = np.array([int(a.get('iscrowd', 0)) for a in truth])
        edge_iou = mask_utils.iou(
            [self._boundary_rle(a, False) for a in detections],
            [self._boundary_rle(a, True) for a in truth], crowd.tolist())
        combined = np.array(region_iou, copy=True)
        regular = crowd == 0
        combined[:, regular] = np.minimum(combined[:, regular], edge_iou[:, regular])
        # Only this image/category needs these potentially large RLEs.
        self._boundary_cache.clear()
        return combined


def load_ground_truth(path):
    coco = COCO(str(path))
    coco.dataset.setdefault('info', {})
    for annotation in coco.dataset.get('annotations', []):
        annotation.setdefault('iscrowd', 0)
    return coco


def prediction_coco(coco_gt, predictions):
    """Normalize RLE/polygon results and force prediction area to mask area."""
    normalized = []
    for index, item in enumerate(predictions):
        image_id, category_id = int(item['image_id']), int(item['category_id'])
        if image_id not in coco_gt.imgs or category_id not in coco_gt.cats:
            raise ValueError(f'Prediction {index}: unknown image/category ID. '
                             'Use the original COCO IDs, not remapped labels.')
        score = float(item['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Prediction {index}: invalid confidence score.')
        if 'segmentation' not in item:
            raise ValueError(f'Prediction {index}: instance segmentation is required.')
        image = coco_gt.imgs[image_id]
        height, width = image['height'], image['width']
        segment = copy.deepcopy(item['segmentation'])
        if isinstance(segment, list):
            if not segment:
                raise ValueError(f'Prediction {index}: empty polygon list; use an empty RLE mask.')
            segment = mask_utils.merge(mask_utils.frPyObjects(segment, height, width))
        elif isinstance(segment, dict):
            if list(segment.get('size', [])) != [height, width]:
                raise ValueError(f'Prediction {index}: mask size differs from original image.')
            if isinstance(segment['counts'], list):
                segment = mask_utils.frPyObjects(segment, height, width)
        else:
            raise ValueError(f'Prediction {index}: unsupported mask format.')
        normalized.append(dict(image_id=image_id, category_id=category_id,
                               score=score, segmentation=segment))
    if normalized:
        # Omitting bbox prevents COCO.loadRes from using box area for masks.
        return coco_gt.loadRes(normalized)
    empty = COCO()
    empty.dataset = dict(images=copy.deepcopy(coco_gt.dataset['images']),
                         categories=copy.deepcopy(coco_gt.dataset['categories']),
                         annotations=[], info={})
    empty.createIndex()
    return empty


def summarize_ap(evaluator):
    precision = evaluator.eval['precision']
    params = evaluator.params
    maximum = list(params.maxDets).index(100)

    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean() * 100) if valid.size else None

    all_area = list(params.areaRngLbl).index('all')
    result = {'AP': mean(precision[:, :, :, all_area, maximum])}
    for threshold, label in ((0.5, 'AP50'), (0.75, 'AP75')):
        result[label] = mean(precision[np.isclose(params.iouThrs, threshold), :, :, all_area, maximum])
    for group, label in (('small', 'APs'), ('medium', 'APm'), ('large', 'APl')):
        result[label] = mean(precision[:, :, :, list(params.areaRngLbl).index(group), maximum])
    result['per_class'] = {
        evaluator.cocoGt.cats[int(category)]['name']: mean(precision[:, :, i, all_area, maximum])
        for i, category in enumerate(params.catIds)}
    return result


def evaluate_predictions(coco_gt, predictions, *, dilation_ratio=0.02, image_ids=None):
    coco_dt = prediction_coco(coco_gt, predictions)
    results = {}
    for label in ('mask', 'boundary'):
        truth = copy.deepcopy(coco_gt)
        detected = copy.deepcopy(coco_dt)
        evaluator = (COCOeval(truth, detected, 'segm') if label == 'mask' else
                     BoundaryCOCOeval(truth, detected, dilation_ratio))
        evaluator.params.maxDets = [1, 10, 100]
        if image_ids is not None:
            evaluator.params.imgIds = list(image_ids)
        print(f'\n--- {label.upper()} AP, IoU 0.50:0.05:0.95 ---', flush=True)
        evaluator.evaluate()
        evaluator.accumulate()
        results[label] = summarize_ap(evaluator)
        # Release RLEs/IoU matrices before the next evaluator.
        del evaluator
    return results
