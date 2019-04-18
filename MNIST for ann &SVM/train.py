# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:58:08 2019

@author: 37112
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection  import train_test_split
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import struct
import time

train_images_file = 'train-images.idx3-ubyte'
train_labels_file = 'train-labels.idx1-ubyte'
test_images_file = 't10k-images.idx3-ubyte'
test_labels_file = 't10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):

    data = open(idx3_ubyte_file, 'rb').read()

    # header details，are magic、image number、h、w 
    offset = 0
    magic_num, image_num, row_num, col_num = struct.unpack_from('>iiii', data, offset)
#    print ('magic:%d, image number: %d, image size: %d*%d' % (magic_num, image_num, row_num, col_num))

    image_size = row_num * col_num
    offset += struct.calcsize('>iiii')
#    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   
    images = np.empty((image_num, row_num*col_num),np.uint8)
    for i in range(image_num):
        images[i] = np.array(struct.unpack_from(fmt_image, data, offset)).reshape((row_num*col_num))
        offset += struct.calcsize(fmt_image)
    return images

def sample(train_images,train_labels):
    train_class_image = [[] for i in range(10)]
    train_class_label = [i for i in range(10)] 
#    for i in range(10):
    for i in range(len(train_labels)):
#        print("train_labels[i] is:", train_labels[i])
        if train_labels[i] == 0:
            train_class_image[0].append(train_images[i])
        elif train_labels[i] == 1:
            train_class_image[1].append(train_images[i])
        elif train_labels[i] == 2:
            train_class_image[2].append(train_images[i])
        elif train_labels[i] == 3:
            train_class_image[3].append(train_images[i])
        elif train_labels[i] == 4:
            train_class_image[4].append(train_images[i])
        elif train_labels[i] == 5:
            train_class_image[5].append(train_images[i])
        elif train_labels[i] == 6:
            train_class_image[6].append(train_images[i])
        elif train_labels[i] == 7:
            train_class_image[7].append(train_images[i])
        elif train_labels[i] == 8:
            train_class_image[8].append(train_images[i])
        elif train_labels[i] == 9:
            train_class_image[9].append(train_images[i])
    return train_class_image, train_class_label

def decode_idx1_ubyte(idx1_ubyte_file):

    data = open(idx1_ubyte_file, 'rb').read()

    # header information，are magic and label
    offset = 0
    magic_num, image_num = struct.unpack_from('>ii', data, offset)
#    print ('magic:%d, image number: %d' % (magic_num, image_num))

    offset += struct.calcsize('>ii')
    fmt_image = '>B'
    labels = np.empty(image_num,dtype = np.int)
    for i in range(image_num):
        labels[i] = struct.unpack_from(fmt_image, data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_images(file):

    return decode_idx3_ubyte(file)

def load_labels(file):

    return decode_idx1_ubyte(file)
    
def run():
    start = time.time()
    train_images = load_images(train_images_file) #(row_num*col_num,image_num)
    train_labels = load_labels(train_labels_file)
    train_class_image, train_class_label = sample(train_images,train_labels)
    train_image, val_images, train_label, val_labels = train_test_split(train_images, train_labels, test_size=0.3,random_state=30)
    train_image_3 = []
    train_label_3 = []
    model = "SVM"
    exp = 3
    if exp == 2:
        train_images = train_images[:48000]
        train_labels = train_labels[:48000]
    if exp == 3:
        for i in range(10):
            if i == 0:
                for j in range(20):
                    train_image_3.append(train_class_image[i][j])
                    train_label_3.append(train_class_label[i])
            else:
                for j in range(5000):
                    train_image_3.append(train_class_image[i][j])
                    train_label_3.append(train_class_label[i])
        train_images = train_image_3
        train_labels = train_label_3
    if model == "SVM": 
#        scaler = StandardScaler() 
#        scaler.fit(train_images)  
#        train_images= scaler.transform(train_images)  
#        clf = svm.SVC(gamma=0.000686, C=25.6, cache_size = 2000)
        clf = svm.SVC(kernel = 'poly')
        _svm = clf.fit(train_images, train_labels)
        joblib.dump(_svm,'svm6.model')
        
    elif model == "ANN":
        scaler = StandardScaler() 
        scaler.fit(train_images)  
        train_images= scaler.transform(train_images)   
        clf = MLPClassifier(solver='adam', activation='relu', alpha=0.0001, hidden_layer_sizes=(130,100,100,10), random_state=1, verbose=10)
        ann = clf.fit(train_images, train_labels)
        joblib.dump(ann,'ann.model')


    finish = time.time()
    print("time use :", finish - start)
    
if __name__ == '__main__':
    run()

