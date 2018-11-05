# -*- coding: utf-8 -*-
import os
dir = "/media/myharddisk/sunhow/"
doc = open("all_files.txt",'w')
for root, dirs, files in os.walk(dir):
    for file in files:
        #print(file)
        if file.split('.')[-1]  in ['jpg', 'JPG']:
            print(os.path.join(root,file), file=doc)
doc.close()
lastLine = ""
listflie = open("file_list.txt","w")
classnum = -1
with open("all_files.txt") as f:
    line = f.readline()
    while line:
        thisline = line.split("/")[-3]+line.split("/")[-2]

        if lastLine==thisline:
            listflie.write(line.split("\n")[0]+"\t"+str(classnum)+"\n")
        else:
            classnum += 1
            lastLine = thisline
            listflie.write(line.split("\n")[0]+"\t"+str(classnum)+"\n")
        line = f.readline()

f.close()
listflie.close()