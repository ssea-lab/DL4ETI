import numpy as np
import random

def creat_list(path):
    lists = [[] for i in range(4)]
    with open(path) as f:
        line = f.readline()
        while line:
            # print(line)
            classnum = int(line.split("\t")[1])
            if classnum == 0 or classnum == 2 or classnum == 3:
                lists[0].append(line.split("\t")[0])
            if classnum == 4 or classnum == 6:
                lists[1].append(line.split("\t")[0])
            if classnum == 1:
                lists[2].append(line.split("\t")[0])
            if classnum == 5:
                lists[3].append(line.split("\t")[0])

            line = f.readline()
    f.close()
    return np.array(lists)

list = creat_list("file_list.txt")
random_list = open("four_random_list.txt","w")
for c in range(len(list)):
    random.shuffle(list[c])
    for item in list[c]:
        random_list.write(str(item)+"\t"+str(c)+"\n")
