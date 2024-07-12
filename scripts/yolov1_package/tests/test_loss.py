import unittest
import torch
from yolov1.loss import yolo_loss

class TestLoss(unittest.TestCase):
    def test_yolo_loss(self):
        y_true = torch.randn(1, 7 * 7 * 30)
        y_pred = torch.randn(1, 7 * 7 * 30)
        loss = yolo_loss(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0)  # Cambiado .item() por .numpy()

if __name__ == '__main__':
    unittest.main()
