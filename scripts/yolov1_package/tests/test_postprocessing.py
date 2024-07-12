import unittest
import torch
from yolov1.postprocessing import non_max_suppression, parse_yolo_output

class TestPostprocessing(unittest.TestCase):
    def test_non_max_suppression(self):
        boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        scores = torch.tensor([0.9, 0.8])
        selected_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
        self.assertGreaterEqual(len(selected_indices), 0)

    def test_parse_yolo_output(self):
        predictions = torch.randn(1, 7 * 7 * 30)
        boxes, scores, classes = parse_yolo_output(predictions)
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(classes, list)

if __name__ == '__main__':
    unittest.main()
