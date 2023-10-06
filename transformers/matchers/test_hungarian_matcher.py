import unittest

import torch

from transformers.matchers.hungarian_matcher import HungarianMatcher
from transformers.utils.datasets.detection.toy_detection_dataset import ToyDetectionDataset


class TestHungarianMatcher(unittest.TestCase):
    def test_matching_simple(self):
        num_obj = 10
        dataset = ToyDetectionDataset(100, num_obj, 10)
        img, gt_boxes = dataset[0]
        logit = torch.ones((1, num_obj, 2))
        logit[:, :, 0] = 0.25

        matcher = HungarianMatcher()
        pred_logits = logit
        pred_boxes = gt_boxes.unsqueeze(1)

        target_classes = [torch.full((num_obj,), 1)]
        gt_boxes_ = [gt_boxes.flip(0)]

        out = matcher.forward(pred_logits, pred_boxes, target_classes, gt_boxes_)

        from_idx, to_idx = out[0]
        self.assertTrue((torch.arange(10) == from_idx).all())
        self.assertTrue((torch.arange(10).flip(0) == to_idx).all())

    def test_matching_complex(self):
        self.assertTrue(False)
