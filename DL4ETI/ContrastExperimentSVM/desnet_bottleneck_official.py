from keras.applications.densenet import DenseNet121,preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
import numpy as np
'''
input: four_random_list.txt
output: desnet_bottleneck.npy
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


input_tensor = Input(shape=(224, 224, 3))
base_model = DenseNet121(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)


all_bottleneck = []
for i in range(len(path_list)):

    tmp_data = image.load_img(path_list[i], target_size=(224, 224))
    input = image.img_to_array(tmp_data)  # x.shape: (224, 224, 3)
    input = np.expand_dims(input, axis=0)  # x.shape: (1, 224, 224, 3)
    input = preprocess_input(input)

    output = model.predict(input)
    print(i)
    print(output)
    all_bottleneck.append(output)


np.save("desnet_bottleneck.npy",np.asarray(all_bottleneck))
