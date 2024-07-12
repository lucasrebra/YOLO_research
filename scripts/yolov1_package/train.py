# -*- coding: utf-8 -*-

from yolov1.model import YOLOv1
from yolov1.preprocessing import create_dataset
from yolov1.loss import yolo_loss
from yolov1.postprocessing import non_max_suppression, parse_yolo_output

def main():
    # Example usage
    input_shape = (448, 448, 3)
    yolo = YOLOv1(input_shape)

    # Prepare your datasets
    filenames = [...]  # List of image file paths
    labels = [...]  # Corresponding list of labels

    # Create dataset
    dataset = create_dataset(filenames, labels)

    # Compile the model
    yolo.compile_model(optimizer='adam', loss=yolo_loss)

    # Split the dataset into training and validation
    train_dataset = dataset.take(8000)
    val_dataset = dataset.skip(8000).take(2000)

    # Train the model
    yolo.train(train_dataset, val_dataset, epochs=50, steps_per_epoch=1000, validation_steps=100)

    # Evaluate the model
    yolo.evaluate(val_dataset)

    # Make predictions
    test_images = [...]  # List of preprocessed test images
    predictions = yolo.predict(test_images)

    # Postprocessing
    boxes, scores, classes = parse_yolo_output(predictions)
    selected_boxes = non_max_suppression(boxes, scores)

if __name__ == "__main__":
    main()
