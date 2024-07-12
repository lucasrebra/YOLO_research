import unittest
import torch
from yolov1.model import YOLOv1

class TestYOLOv1(unittest.TestCase):
    def test_model_creation(self):
        model = YOLOv1(input_shape=(448, 448, 3))
        self.assertIsNotNone(model)

    def test_model_forward_pass(self):
        model = YOLOv1(input_shape=(448, 448, 3))
        input_tensor = torch.randn(1, 3, 448, 448)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 7 * 7 * 30))

if __name__ == '__main__':
    unittest.main()
