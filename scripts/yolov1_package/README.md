
# YOLOv1 Package

This package implements the YOLOv1 object detection model using TensorFlow. It includes components for building, training, and evaluating the model, as well as preprocessing and postprocessing functions.

## Contents

- `yolov1/`
  - `__init__.py`: Initialization file that imports key components of the package.
  - `model.py`: Contains the `YOLOv1` class and `ConvBlock` class for building the YOLOv1 model.
  - `preprocessing.py`: Contains functions for preprocessing images.
  - `postprocessing.py`: Contains functions for postprocessing, including non-max suppression (NMS).
  - `loss.py`: Contains the YOLOv1 loss function.
- `train.py`: Main script for training the YOLOv1 model.
- `setup.py`: Setup script for installing the package.

## Installation

To install the package, navigate to the directory containing `setup.py` and run:

```sh
pip install .
```

## Training the Model on a Custom Dataset

### Prepare the Dataset

1. Organize your dataset with images and labels.
2. Create a list of image file paths and their corresponding labels.

### Create the Dataset

Use the `create_dataset` function from the `preprocessing` module to create a TensorFlow dataset.

### Compile the Model

Use the `compile_model` method from the `YOLOv1` class, specifying the optimizer and loss function.

### Train the Model

Use the `train` method from the `YOLOv1` class, specifying the training and validation datasets, the number of epochs, steps per epoch, and validation steps.

### Example Workflow

1. **Import the necessary components:**

```python
from yolov1.model import YOLOv1
from yolov1.preprocessing import create_dataset
from yolov1.loss import yolo_loss
```

2. **Prepare your datasets:**

```python
filenames = [...]  # List of image file paths
labels = [...]  # Corresponding list of labels
dataset = create_dataset(filenames, labels)
```

3. **Initialize and compile the model:**

```python
input_shape = (448, 448, 3)
yolo = YOLOv1(input_shape)
yolo.compile_model(optimizer='adam', loss=yolo_loss)
```

4. **Split the dataset into training and validation:**

```python
train_dataset = dataset.take(8000)
val_dataset = dataset.skip(8000).take(2000)
```

5. **Train the model:**

```python
yolo.train(train_dataset, val_dataset, epochs=50, steps_per_epoch=1000, validation_steps=100)
```

6. **Evaluate the model:**

```python
yolo.evaluate(val_dataset)
```

7. **Make predictions and postprocess:**

```python
test_images = [...]  # List of preprocessed test images
predictions = yolo.predict(test_images)

from yolov1.postprocessing import non_max_suppression, parse_yolo_output

boxes, scores, classes = parse_yolo_output(predictions)
selected_boxes = non_max_suppression(boxes, scores)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
