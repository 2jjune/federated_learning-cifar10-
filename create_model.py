import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from keras.regularizers import l2

from keras import regularizers
from keras import layers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

def create_model():
    tmp = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20, activation='softmax')
    ])

    # tmp = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', input_shape=(32,32,3), kernel_regularizer = 'l2'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    # tmp = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'elu', input_shape=(32,32,3), padding = 'same', kernel_regularizer = l2(0.0001)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='elu', padding='same', kernel_regularizer = l2(0.0001)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Dropout(0.2),
    #
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same', kernel_regularizer = l2(0.0001)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same', kernel_regularizer=l2(0.0001)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Dropout(0.3),
    #
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_regularizer=l2(0.0001)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_regularizer=l2(0.0001)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Dropout(0.4),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    # tmp = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', padding = 'same', input_shape=(32,32,3)),
    #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', padding = 'same'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Dropout(0.25),
    #
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu', padding = 'same'),
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu', padding = 'same'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Dropout(0.25),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    # tmp = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu',
    #                            input_shape=(32, 32, 3)),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    #     tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Dropout(0.25),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(1000, activation='relu'),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])

    return tmp