import random
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
from keras.applications.inception_v3 import InceptionV3
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
import tools
import gc
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from keras.models import Model
import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import initializers
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

                if "分泌性子宫内膜"in p or "增生性子宫内膜"in p or "月经期"in p:
                    data.append(p)
                    label.append(0)
                if "子宫内膜复杂性增生" in p or "子宫内膜单纯性增生" in p:
                    data.append(p)
                    label.append(1)
                if "子宫内膜息肉" in p:
                    data.append(p)
                    label.append(2)
                if "子宫内膜癌" in p:
                    data.append(p)
                    label.append(3)

            line = f.readline()
    f.close()
    return data,label

base_model = ResNet50(input_tensor=Input(shape=(224, 224, 3)), weights=None, include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('/media/myharddisk/sunhow/ResNet50/tmp/3-weights.116-1.0000-2.3780-0.6319.h5')

test_data_path, test_label = read_data_label("tmp.txt")
test_data = []
print(test_data,test_label)
for test_path in test_data_path:
    test_data.append(tools.read_image(test_path, 224, 224, True))

batch_size = 1

#计算confusion matrix
all_y_pred = []
all_y_true = []
for test_data_batch, test_label_batch in batch_generator_confusion_matrix(np.array(test_data),np.array(test_label), batch_size, True, 4):
    y_pred = model.predict(test_data_batch, batch_size)
    y_true = test_label_batch
    for y_p in y_pred:
        all_y_pred.append(np.where(y_p == max(y_p))[0][0])

    for y_t in y_true:
        all_y_true.append(np.where(y_t == max(y_t))[0][0])
confusion = confusion_matrix(y_true=all_y_true,y_pred=all_y_pred)
print(confusion)
f = open('cm.txt','w')
f.write(str(all_y_true)+"\n")
f.write(str(all_y_pred)+"\n")
f.write(str(confusion)+'\n')
f.close()
gc.enable()
