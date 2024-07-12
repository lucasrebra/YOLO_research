# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.bn = BatchNormalization()
        self.activation = LeakyReLU(alpha=0.1) if activation else None

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

class YOLOv1:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # Block 1
        x = ConvBlock(64, (7, 7), strides=2, padding='same')(inputs)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

        # Block 2
        x = ConvBlock(192, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

        # Block 3
        x = ConvBlock(128, (1, 1), strides=1, padding='same')(x)
        x = ConvBlock(256, (3, 3), strides=1, padding='same')(x)
        x = ConvBlock(256, (1, 1), strides=1, padding='same')(x)
        x = ConvBlock(512, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

        # Block 4
        for _ in range(4):
            x = ConvBlock(256, (1, 1), strides=1, padding='same')(x)
            x = ConvBlock(512, (3, 3), strides=1, padding='same')(x)
        x = ConvBlock(512, (1, 1), strides=1, padding='same')(x)
        x = ConvBlock(1024, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D((2, 2), strides=2, padding='same')(x)

        # Block 5
        for _ in range(2):
            x = ConvBlock(512, (1, 1), strides=1, padding='same')(x)
            x = ConvBlock(1024, (3, 3), strides=1, padding='same')(x)
        x = ConvBlock(1024, (3, 3), strides=1, padding='same')(x)
        x = ConvBlock(1024, (3, 3), strides=2, padding='same')(x)

        # Block 6
        x = ConvBlock(1024, (3, 3), strides=1, padding='same')(x)
        x = ConvBlock(1024, (3, 3), strides=1, padding='same')(x)

        # Fully connected layers
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(7 * 7 * 30, activation='linear')(x)

        model = Model(inputs, x)
        return model

    def compile_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps):
        self.model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, validation_steps=validation_steps)

    def evaluate(self, val_dataset):
        return self.model.evaluate(val_dataset)

    def predict(self, images):
        return self.model.predict(images)
