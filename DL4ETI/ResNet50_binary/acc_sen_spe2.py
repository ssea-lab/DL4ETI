import random
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, BatchNormalization
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard,CSVLogger
import gc
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from keras.models import Model
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import initializers
from keras.models import load_model
import tools
from sklearn import metrics
import os


def batch_generator_confusion_matrix(all_data, all_label, batch_size, shuffle, class_num = 4):
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


def conduct(fileName,test_data_path,test_label):
    base_model = ResNet50(input_tensor=Input(shape=(224, 224, 3)), weights=None, include_top=False)

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(4, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    model.load_weights(rootPath+fileName)

    test_data = []
    print(test_data,test_label)
    for test_path in test_data_path:
        test_data.append(tools.read_image(test_path, 224, 224, True))

    batch_size = 1

    all_y_pred = []
    all_y_true = []
    for test_data_batch, test_label_batch in batch_generator_confusion_matrix(np.array(test_data),np.array(test_label), batch_size, True, 2):
        y_pred = model.predict(test_data_batch, batch_size)
        y_true = test_label_batch
        for y_p in y_pred:
            all_y_pred.append(np.where(y_p == max(y_p))[0][0])

        for y_t in y_true:
            all_y_true.append(np.where(y_t == max(y_t))[0][0])
    confusion = confusion_matrix(y_true=all_y_true,y_pred=all_y_pred)

    acc = metrics.accuracy_score(y_true=all_y_true,y_pred=all_y_pred)
    print(confusion)
    print(acc)
    sens = confusion[1][1] / (confusion[1][0] + confusion[1][1])
    spec = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    accfile = open("acc.txt", "a+")
    accfile.write(str(acc) + '\t' + str(sens) + "\t" + str(spec) + "\n")
    accfile.close()
    gc.enable()

rootPath = "/media/myharddisk/sunhow/ResNet50_binary/tmp2/"
path = []
label = []
for root, dirs, sigle_files in os.walk(rootPath):
    next = -1
    print(sigle_files)
    for files in sigle_files:
        for i in range(10):
            if str(i)+"-weights" in files:
                with open("records.txt") as f:
                    line = f.readline()
                    while line:
                        if str(i)+"fold" in line:
                            next = 0
                            line = f.readline()
                        if next==0:
                            next = 1
                            line = f.readline()
                        if next == 1:
                            thisline = str(line)
                            new = thisline.replace('[', '').replace(']', '').replace("'", '').replace(" /", "/")
                            print(new)
                            next = -1
                            path = new.split(",")
                            label = []
                            for i in range(len(path)):
                                p = path[i]
                                if "NELuteal" in p or "NEFollicular" in p or "NEMenstrual" in p:
                                    label.append(0)
                                if "EHSimple" in p or "EHComplex" in p:
                                    label.append(1)
                                if "EP" in p:
                                    label.append(2)
                                if "EA" in p:
                                    label.append(3)

                            conduct(files,path,label)
                            break
                        if next== -1:
                            line = f.readline()


                f.close()

