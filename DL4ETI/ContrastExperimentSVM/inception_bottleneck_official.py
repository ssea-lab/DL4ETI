from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
'''
input: four_random_list.txt
output: inception_bottleneck.npy
'''
def create_list(filepath):
    path_list = []
    label_list = []
    with open(filepath) as f:
        line = f.readline()
        while line:
            path = line.split('\t')[0]
            label = line.split('\t')[1].split('\n')[0]
            path_list.append(path)
            label_list.append(label)
            line = f.readline()
    f.close()

    return path_list,label_list


path_list, label_list = create_list('four_random_list.txt')



input_tensor = Input(shape=(299, 299, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

all_bottleneck = []
for i in range(len(path_list)):

    tmp_data = image.load_img(path_list[i], target_size=(299, 299))
    input = image.img_to_array(tmp_data)
    input = np.expand_dims(input, axis=0)
    input = preprocess_input(input)

    output = model.predict(input)
    print(i)
    print(output)
    all_bottleneck.append(output)


np.save("inception_bottleneck.npy",np.asarray(all_bottleneck))
