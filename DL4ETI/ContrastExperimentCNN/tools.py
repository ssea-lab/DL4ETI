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
from PIL import Image,ImageEnhance

def read_image(imagePath, width=600, height=600, normalization = True):
    img = io.imread(imagePath.split('\n')[0])
    if normalization == True:
        imageData = transform.resize(img,(width, height, 3),order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None)
        imageData = np.transpose(imageData,(2,0,1))
        imageData[0] = preprocessing.scale(imageData[0])
        imageData[1] = preprocessing.scale(imageData[1])
        imageData[2] = preprocessing.scale(imageData[2])
        imageData = np.transpose(imageData,(1,2,0))
        # imageData = transform.resize(img,(width, height,3))
    else:
        imageData = transform.resize(img,(width, height,3))
    return imageData

def dataAugmentation(input_array, flip_left_right_rate=0.5, flip_top_bottom_rate=0.5):
    # array转Image
    if (random.random() < flip_left_right_rate):
        # 左右翻转
        input_array = np.transpose(input_array,(2,0,1))# 源w,h,3- > 3,w,h
        input_array[0] = np.flip(input_array[0], 1)
        input_array[1] = np.flip(input_array[1], 1)
        input_array[2] = np.flip(input_array[2], 1)
        input_array = np.transpose(input_array, (1,2,0))#源3, w, h -> w, h, 3
    if (random.random() < flip_top_bottom_rate):
        # 上下翻转
        input_array = np.transpose(input_array, (2, 0, 1))  # 源w,h,3- > 3,w,h
        input_array[0] = np.flip(input_array[0], 0)
        input_array[1] = np.flip(input_array[1], 0)
        input_array[2] = np.flip(input_array[2], 0)
        input_array = np.transpose(input_array, (1, 2, 0))  # 源3, w, h -> w, h, 3

    return np.array(input_array)

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

#添加data以及label
def add_array(arr1,arr2):
    for a in arr2:
        arr1.append(a)
    return arr1


#根据给定的key读取相应的文件名以及打上标签
def create_data_label(dic,key,label_id):
    data_path = []
    label = []
    for i in key:
        for v in dic[i]:
            label.append(label_id)
            data_path.append(v)
    return data_path, label

#输出是某一类 train 和 test的文件路径的key
def slice_train_test(array, i, K):
    length = len(array)
    step = math.floor(length / K)
    if step == 0:
        step = 1
    print('总长度%d为步长%d' %(length, step))
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


'''
generator的输入是所有train/test的 path,label
输出 batch 大小的 np.array data,label
'''
def batch_generator(all_data, all_label, batch_size, shuffle, class_num = 6,train = True):
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
                tmp_data = all_data[di]
                if train:
                    tmp_data = dataAugmentation(tmp_data)
                data.append(tmp_data)

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
            tmp_data = all_data[di]

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

#返回np.array中最大数字的下标
#如果arr是list 记得要转为array
def arr_max_index(arr):
    return np.where(arr == max(arr))[0][0]

