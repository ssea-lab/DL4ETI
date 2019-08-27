#coding=utf-8
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
from keras.applications.inception_v3 import InceptionV3
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
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras import initializers


def creat_list(path):
    lists = [[] for i in range(4)]
    with open(path) as f:
        line = f.readline()
        while line:
            # print(line)
            classnum = int(line.split("\t")[1])
            lists[classnum].append(line.split("\t")[0])
            line = f.readline()
    f.close()
    return np.array(lists)


def cross_validation(data, K, epoch, class_num, batch_size):
    category = len(data)
    print(category)
    print("=========================")
    # if shuffle:
    #     for c in range(category):
    #         random.shuffle(data[c])
    for i in range(0,K):
        print("%d fold" % i)
        train_data_path = []
        train_label = []
        test_data_path = []
        test_label = []
        for c in range(category):
            part_trian_data_path, part_test_data_path = tools.slice_train_test(data[c], i, K)

            for train_len in range(len(part_trian_data_path)):
                train_data_path.append(part_trian_data_path[train_len])
                train_label.append(c)

            for test_len in range(len(part_test_data_path)):
                test_data_path.append(part_test_data_path[test_len])
                test_label.append(c)

        print(len(train_data_path), len(train_label))
        print(len(test_data_path), len(test_label))

        record = open('records.txt', 'a+')
        record.write("%d fold\n" % i)
        record.write(str(train_data_path) + '\n')
        record.write(str(test_data_path) + '\n')
        record.close()


        '''
        train_data_path = []
        train_label = []
        test_data_path = []
        test_label = []
        '''

        train_data = []
        test_data = []
        for train_path in train_data_path:
            train_data.append(tools.read_image(train_path, 299, 299, True))
        for test_path in test_data_path:
            test_data.append(tools.read_image(test_path, 299, 299, True))

        Network_config(class_num = class_num, epoch = epoch, initial_epoch = 0, batch_size = batch_size,
                       train_data=train_data, train_label=train_label,
                       test_data=test_data, test_label=test_label, fold = i)
    return

def Network_config(class_num=4, epoch=200, initial_epoch=0, batch_size=32,
                     train_data=None, train_label=None,
                     test_data=None, test_label=None, fold=0):
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000)
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    K.set_learning_phase(1)
    base_model = InceptionV3(input_tensor=Input(shape=(299, 299, 3)), weights='imagenet', include_top=False)

    x = base_model.output
    # K.set_learning_phase(1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(class_num, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in (base_model.layers):
        layer.trainable = False
        if layer.name.startswith('bn') or 'bn' in layer.name:
            layer.call(layer.input, training=False)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.categorical_accuracy])


    tools.create_directory('./tmpinception/')
    weights_file = './tmpinception/' + str(fold)+'-weights.{epoch:02d}-{categorical_accuracy:.4f}-{val_loss:.4f}-{val_categorical_accuracy:.4f}.h5'
    csv_file = './tmpinception/record.csv'
    lr_reducer = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5,
                                   cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_categorical_accuracy', min_delta=1e-4, patience=50)

    model_checkpoint = ModelCheckpoint(weights_file, monitor='val_categorical_accuracy', save_best_only=True,
                                       verbose=2,
                                       save_weights_only=True, mode='max')
    tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=8, write_graph=True,
                              write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    CSV_record = CSVLogger(csv_file, separator=',', append=True)

    callbacks = [lr_reducer, early_stopper, model_checkpoint, tensorboard, CSV_record]
    gc.disable()
    model.fit_generator(
        generator=tools.batch_generator(np.array(train_data), np.array(train_label), batch_size, True, class_num),
        steps_per_epoch=int(len(train_label)/batch_size)-1,
        max_q_size=20,
        initial_epoch=initial_epoch,
        epochs=epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=tools.batch_generator(np.array(test_data), np.array(test_label), batch_size, True, class_num),
        validation_steps=int(len(test_label)/batch_size)-1,
        class_weight='auto')


    all_y_pred = []
    all_y_true = []
    for test_data_batch, test_label_batch in tools.batch_generator_confusion_matrix(np.array(test_data),np.array(test_label), batch_size, True, class_num):
        y_pred = model.predict(test_data_batch, batch_size)
        y_true = test_label_batch
        for y_p in y_pred:
            all_y_pred.append(np.where(y_p == max(y_p))[0][0])

        for y_t in y_true:
            all_y_true.append(np.where(y_t == max(y_t))[0][0])
    confusion = confusion_matrix(y_true=all_y_true,y_pred=all_y_pred)
    print(confusion)
    f = open('confusion_matrix.txt','a+')
    f.write(str(all_y_true)+"\n")
    f.write(str(all_y_pred)+"\n")
    f.write(str(confusion)+'\n')
    f.close()
    gc.enable()


def main():
    data = creat_list('four_random_list.txt')
    cross_validation(data, K=10, epoch=200, class_num=4, batch_size=32)
main()