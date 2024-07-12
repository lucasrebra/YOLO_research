# -*- coding: utf-8 -*-

import tensorflow as tf

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=iou_threshold)
    selected_boxes = tf.gather(boxes, indices)
    return selected_boxes

def parse_yolo_output(predictions):
    # Implement parsing logic
    boxes = [...]  # Extracted boxes
    scores = [...]  # Extracted scores
    classes = [...]  # Extracted classes
    return boxes, scores, classes
