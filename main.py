import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import set_data
import set_epoch_num
import create_model
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


model = []
epoch = set_epoch_num.set_epoch
set_num = set_epoch_num.set_num

for i in range(set_num):
    model.append(create_model.create_model())

new_model = create_model.create_model()

for i in range(set_num):
    model[i].load_weights("{}_epoch {}.h5".format(epoch, i+1))

weights = []
for i in range(set_num):
    weights.append(model[i].get_weights())
print('weights 갯수 : ',len(weights))


new_weights_avg = list()
new_weights_median = list()
new_weights_custom = list()

for weights_list_tuple in zip(*weights):
    new_weights_avg.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        # np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
    )
    new_weights_median.append(
        np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
    )
    new_weights_custom.append(
        np.array([(np.max(np.array(w), axis=0)+np.min(np.array(w), axis=0))/2 for w in zip(*weights_list_tuple)])
    )

#모델 evaluate
new_model.set_weights(new_weights_avg)
new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
avg_test_loss, avg_test_acc = new_model.evaluate(set_data.test_images, set_data.test_labels)

new_model.set_weights(new_weights_median)
new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
test_loss, test_acc = new_model.evaluate(set_data.test_images, set_data.test_labels)

new_model.set_weights(new_weights_custom)
new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
custom_test_loss, custom_test_acc = new_model.evaluate(set_data.test_images, set_data.test_labels)

print('avg 테스트 정확도: {}, loss: {}'.format(avg_test_acc,avg_test_loss))
print('median 테스트 정확도: {}, loss: {}'.format(test_acc,test_loss))
print('custom 테스트 정확도: {}, loss: {}'.format(custom_test_acc,custom_test_loss))