# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:41:45 2019

@author: 37112
"""

import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import struct
import time

test_images_file = 't10k-images.idx3-ubyte'
test_labels_file = 't10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):

    data = open(idx3_ubyte_file, 'rb').read()

    # header details，are magic、image number、h、w 
    offset = 0
    magic_num, image_num, row_num, col_num = struct.unpack_from('>iiii', data, offset)

    image_size = row_num * col_num
    offset += struct.calcsize('>iiii')
    fmt_image = '>' + str(image_size) + 'B'   
    images = np.empty((image_num, row_num*col_num),np.uint8)
    for i in range(image_num):
        images[i] = np.array(struct.unpack_from(fmt_image, data, offset)).reshape((row_num*col_num))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):

    data = open(idx1_ubyte_file, 'rb').read()

    # header information，are magic and label
    offset = 0
    magic_num, image_num = struct.unpack_from('>ii', data, offset)

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

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--model', type=str, dest='mod',
                        help='the filename whicn needs to be test', default='ANN')
    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    
    args = parse_args()
  
    start = time.time()
    test_images = load_images(test_images_file)
    test_labels = load_labels(test_labels_file) 
    exp = 1
    if exp == 2:
        test_images = test_images[:10000]
        test_labels = test_labels[:10000]
    model = args.mod
    if model == "SVM":
#        scaler = StandardScaler() 
#        scaler.fit(test_images)  
#        test_images= scaler.transform(test_images)   
        RF = joblib.load('svm.model')
        
    elif model == "ANN":
        RF = joblib.load('ann.model')
    
    predict_results = RF.predict(test_images)
    print("accuracy_score:",accuracy_score(predict_results, test_labels))
    conf_mat = confusion_matrix(test_labels, predict_results)
    print("conf_mat:")
    print(conf_mat)
    print(classification_report(test_labels, predict_results))

    finish = time.time()
    print("time use :", finish - start)

