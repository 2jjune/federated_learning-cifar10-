import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import create_model
import set_epoch_num
import set_data

set_epoch = set_epoch_num.set_epoch
set_num = set_epoch_num.set_num
print(set_epoch, set_num)

model = []
for i in range(set_num):
    model.append(create_model.create_model())
    # print(model[i])

for i in range(set_num):
    model[i].compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model[i].summary()

for i in range(len(model)):
    # model_i = model[i]
    print()
    print(i,"*******************************")
    model[i].fit(set_data.train_images, set_data.train_labels, batch_size=64, epochs=set_epoch, validation_data=(set_data.test_images,set_data.test_labels))
    model[i].save_weights('{}_epoch {}.h5'.format(set_epoch,i+1))







# 1. MNIST 데이터셋 임포트
# def set_mnist():
#     mnist = tf.keras.datasets.fashion_mnist
#     (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#     training_images = training_images.reshape(60000, 28, 28, 1)
#     training_images = training_images / 255.0
#     test_images = test_images.reshape(10000, 28, 28, 1)
#     test_images = test_images / 255.0
#
#     training_labels = to_categorical(training_labels)
#     test_labels = to_categorical(test_labels)
#     return training_images, training_labels
# 3. 모델 구성
