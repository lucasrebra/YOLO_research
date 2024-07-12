import unittest
from yolov1.preprocessing import create_dataset

class TestPreprocessing(unittest.TestCase):
    def test_create_dataset(self):
        filenames = ['dummy_path1', 'dummy_path2']
        labels = [0, 1]
        dataset = create_dataset(filenames, labels)
        # Verifica que el dataset tenga elementos
        dataset_length = sum(1 for _ in dataset)
        self.assertEqual(dataset_length, 2)

if __name__ == '__main__':
    unittest.main()
