# -*- coding: utf-8 -*-

import tensorflow as tf

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [448, 448])
    return image_resized, label

def create_dataset(filenames, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size).repeat()
    return dataset
