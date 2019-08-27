import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
import tools
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import math
from sklearn.decomposition import PCA

bottleneck_file = 'resnet_bottleneck.npy'
name_label_file = 'four_random_list.txt'
K = 10
category = 4

def create_list(filepath):
    path_list = []
    label_list = []
    with open(filepath,encoding='utf-8') as f:
        line = f.readline()
        while line:
            path = line.split('\t')[0]
            label = line.split('\t')[1].split('\n')[0]
            path_list.append(path)
            label_list.append(label)
            line = f.readline()
    f.close()

    return path_list,label_list

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


#读取data
all_data = np.load(bottleneck_file)

data_list = np.squeeze(all_data)
estimator = PCA(n_components=50)
data_list=estimator.fit_transform(data_list)




print(data_list.shape)

path_list, label_list = create_list(name_label_file)

data =  [[] for i in range(category)]

for i in range(len(label_list)):
    cls = int(label_list[i])
    data[cls].append(data_list[i])



pred_4_acc = []
pred_2_acc = []

#划分data,进行十折
for i in range(K):
    print("%d fold" % i)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for c in range(category):
        part_trian_data_path, part_test_data_path = slice_train_test(data[c], i, K)

        for train_len in range(len(part_trian_data_path)):
            train_data.append(part_trian_data_path[train_len])
            train_label.append(c)

        for test_len in range(len(part_test_data_path)):
            test_data.append(part_test_data_path[test_len])
            test_label.append(c)


    # print(len(train_data))
    # print(len(test_data))
    # print(len(train_data[0]))
    print(len(train_data), len(train_label))
    print(len(test_data), len(test_label))
    # param_grid = {
    #     'C': [0.01,0.1,1.0, 5, 10, 20, 40,100],
    #     'gamma': [0.0001, 0.001, 0.005, 0.01, 0.1, 1, 5, 10],
    # }
    param_grid = {
        'C': [2 ** j for j in range(-5, 15, 2)],
        'gamma': [2 ** j for j in range(3, -15, -2)],
    }
    svc = svm.SVC(kernel='rbf', class_weight='balanced',verbose=False,probability=True)

    # clf = GridSearchCV(svc, param_grid, scoring='f1', n_jobs=10, cv=5)
    clf = GridSearchCV(svc, param_grid, n_jobs=10, cv=3)
    clf.fit(train_data,train_label)
    y_pred = clf.predict(train_data)
    print(classification_report(train_label, y_pred))
    print(confusion_matrix(train_label, y_pred))
    print('===============train-end=============')

    y_pred = clf.predict(test_data)
    print(classification_report(test_label, y_pred))
    print(confusion_matrix(test_label, y_pred))
    


#####################################################################
    svc = svm.SVC(kernel='rbf', class_weight='balanced',
              C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    print('--------best params--------')
    print(clf.best_params_['C'])
    print(clf.best_params_['gamma'])

    model = svc.fit(train_data, train_label)

    print(model)


    pred_label = model.predict(test_data)
    print(classification_report(test_label, pred_label))
    print(confusion_matrix(test_label, pred_label))

    #2-class
    b_test_label = []
    b_pred_label = []

    for l in range(len(test_data)):
        if test_label[l] == 3:
            b_test_label.append(1)
        else:
            b_test_label.append(0)
        if y_pred[l] == 3:
            b_pred_label.append(1)
        else:
            b_pred_label.append(0)

    # print(accuracy_score(b_test_label, b_pred_label))

    pred_4_acc.append(accuracy_score(test_label,pred_label))
    pred_2_acc.append(accuracy_score(b_test_label,b_pred_label))

print(pred_4_acc)
print(pred_2_acc)

print(np.average(np.asarray(pred_4_acc)))
print(np.average(np.asarray(pred_2_acc)))
print(np.std(np.asarray(pred_4_acc)))
print(np.std(np.asarray(pred_2_acc)))

