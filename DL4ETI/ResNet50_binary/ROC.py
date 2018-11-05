import random
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
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
from keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import cross_validation

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

def read_data_label(path):
    data = []
    label = []
    with open(path) as f:
        line = f.readline()
        while line:
            line.replace("[","")
            line.replace("]","")

            path = line.split(",")
            # path =  list(map(str,path))
            for i in range(len(path)):
                p = path[i].replace("'", "").replace("[","").replace("]","")

                if "NELuteal" in p or "NEFollicular" in p or "NEMenstrual" in p:
                    data.append(p)
                    label.append(0)
                if "EHSimple" in p or "EHComplex" in p:
                    data.append(p)
                    label.append(0)
                if "EP" in p:
                    data.append(p)
                    label.append(0)
                if "EA" in p:
                    data.append(p)
                    label.append(1)

            line = f.readline()
    f.close()
    return data,label

base_model = ResNet50(input_tensor=Input(shape=(224, 224, 3)), weights=None, include_top=False)

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('/media/myharddisk/sunhow/ResNet50_binary/tmp/4-weights.71-1.0000-0.6003-0.9132.h5')

test_data_path, test_label = read_data_label("tmp.txt")
test_data = []
print(test_data,test_label)
for test_path in test_data_path:
    test_data.append(tools.read_image(test_path, 224, 224, True))

batch_size = 1

all_y_pred = []
all_y_true = []
y_score=[]
for test_data_batch, test_label_batch in batch_generator_confusion_matrix(np.array(test_data),np.array(test_label), batch_size, True, 2):
    y_pred = model.predict(test_data_batch, batch_size)
    y_true = test_label_batch
    y_score.append(y_pred[0][1])
    for y_p in y_pred:
        all_y_pred.append(np.where(y_p == max(y_p))[0][0])

    for y_t in y_true:
        all_y_true.append(np.where(y_t == max(y_t))[0][0])
confusion = confusion_matrix(y_true=all_y_true,y_pred=all_y_pred)
print(confusion)
fpr, tpr, threshold = roc_curve(all_y_true, y_score,pos_label=1)
roc_auc = auc(fpr, tpr)
print(fpr.tolist())
print(tpr.tolist())
print(roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve (ResNet50)')
plt.legend(loc="lower right")
plt.show()
gc.enable()
