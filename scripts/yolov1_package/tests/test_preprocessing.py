import unittest
from yolov1.preprocessing import create_dataset

class TestPreprocessing(unittest.TestCase):
    def test_create_dataset(self):
        filenames = ['dummy_path1', 'dummy_path2']
        labels = [0, 1]
        dataset = create_dataset(filenames, labels)
        self.assertEqual(len(dataset), 2)

if __name__ == '__main__':
    unittest.main()
