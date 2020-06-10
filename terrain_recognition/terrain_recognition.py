import os

import skimage.color
import skimage.feature
import skimage.io
import skimage.transform
import sklearn.svm
from sklearn.model_selection import train_test_split 

import numpy as np

def get_data():
    X_data = []
    Y_label = []
    cwd = 'F:/Git/terrain_recognition/Terrain8/'
    classes = {'Asphalt','Dirt','Floor','Grass','Gravel','Rock','Sand','Wood chips'}
    for index, name in enumerate(classes):
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            im = skimage.io.imread(img_path)
            hf = skimage.feature.hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

            X_data.append(hf)
            Y_label.append(name)
    return X_data, Y_label


X_data, Y_label = get_data()
X_tr,X_te,Y_tr,Y_te = train_test_split(X_data,Y_label,test_size=0.4, random_state=0)

clf = sklearn.svm.SVC(probability=True)
clf.fit(X_tr, Y_tr)

r = clf.predict(X_te)
s = 0
for i in range(len(r)):
    print(r[i],Y_te[i])
    if r[i] == Y_te[i]:
        s += 1
print('acc:', s / len(r))







   