#coding=utf-8
from skimage import io, transform
import glob,math
import os
import tensorflow as tf
import numpy as np
import time
from sklearn import preprocessing
import random
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD,Adam
import numpy as np
from keras import backend as K

import gc
import psutil

def read_image(imagePath, width=227, height=227, normalization = True):
    img = io.imread(imagePath.split('\n')[0])
    if normalization == True:
        imageData = transform.resize(img,(3, width, height))
        imageData[0] = preprocessing.scale(imageData[0])
        imageData[1] = preprocessing.scale(imageData[1])
        imageData[2] = preprocessing.scale(imageData[2])
        imageData = transform.resize(imageData, (width, height, 3))
    else:
        imageData = transform.resize(img,(width, height, 3))
    return imageData

def read_bottleneck(bottleneckPath):
    bottleneck = []
    bottleneckFile  =  open(bottleneckPath.split('\n')[0])
    content = bottleneckFile.read()
    feature = content.split(',')
    for i in feature[0:-1]:
        bottleneck.append(float(i))
    bottleneck = preprocessing.scale(bottleneck)
    bottleneck = np.asarray(bottleneck)
    return bottleneck

def add_array(arr1,arr2):
    for a in arr2:
        arr1.append(a)
    return arr1


def create_data_label(dic,key,label_id):
    data_path = []
    label = []
    for i in key:
        for v in dic[i]:
            label.append(label_id)
            data_path.append(v)
    return data_path, label

def slice_train_test(array, i, K):
    length = len(array)
    step = math.floor(length / K)
    if step == 0:
        step = 1
    print('length %d step %d' %(length, step))
    train = []
    test = []
    if i==(K-1):
        for x in array[0: (min((step * i), (length - 1)))]:
            train.append(x)
        for x in array[(min((step * i), (length - 1))):]:
            test.append(x)
    else:
        for x in array[0: (min((step * i),(length - 1)))]:
            train.append(x)

        for x in array[(min((step * i),(length - 1))):(min((step * (i + 1),(length - 1))))]:
            test.append(x)

        for x in array[(min((step * (i + 1),(length - 1)))): ]:
            train.append(x)
    return train, test

def batch_generator(all_data, all_label, batch_size, shuffle, class_num = 6):
    assert len(all_data) == len(all_label)
    print(len(all_data))
    if shuffle:
        indices = np.arange(len(all_data))
        random.shuffle(indices)
    while True:
        for start_idx in range(0, len(all_data) - batch_size + 1, batch_size):
            data = []
            labels = []
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            for di in excerpt:
                data.append(all_data[di])

            for li in excerpt:
                cla = all_label[li]
                tmp = [0 for x in range(class_num)]
                tmp[cla] = 1
                labels.append(tmp)

            yield np.array(data),np.array(labels)

            
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def batch_generator_confusion_matrix(all_data, all_label, batch_size, shuffle, class_num = 6):
    assert len(all_data) == len(all_label)
    print(len(all_data))

    if shuffle:
        indices = np.arange(len(all_data))
        random.shuffle(indices)

    for start_idx in range(0, len(all_data) - batch_size + 1, batch_size):
        data = []
        labels = []
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        for di in excerpt:
            data.append(all_data[di])

        for li in excerpt:
            cla = all_label[li]
            tmp = [0 for x in range(class_num)]
            tmp[cla] = 1
            labels.append(tmp)

        yield np.array(data), np.array(labels)

def create_directory(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def arr_max_index(arr):
    return np.where(arr == max(arr))[0][0]

