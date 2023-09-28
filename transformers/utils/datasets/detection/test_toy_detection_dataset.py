import unittest

from transformers.utils.datasets.detection.toy_detection_dataset import ToyDetectionDataset


class TestToyDetectionDataset(unittest.TestCase):
    def test_data(self):
        dataset = ToyDetectionDataset(10, 1, 100)
        x, y = dataset[0]

        self.assertEqual((1, 10, 10), x.shape)
        self.assertEqual((1, 4), y.shape)
        for i in range(4):
            self.assertTrue(0 <= y[0, i] <= 1)
        self.assertEqual(100, len(dataset))
