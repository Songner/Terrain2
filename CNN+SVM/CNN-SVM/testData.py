# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:34:25 2018

@author: dp
"""
import random

import numpy as np
import tensorflow as tf
from sklearn import svm


for file_num in range(10):
    # 在十个随机生成的不相干数据集上进行测试，将结果综合
    print('testing NO.%d dataset.......' % file_num)
    ff = open('digit_train_' + file_num.__str__() + '.data')
    rr = ff.readlines()
    x_test2 = []
    y_test2 = []
    
    #数据表示   
    tmp=map(int, map(float, rr[114].split(' ')[:256]))
    print(list(tmp))
    print('\n\n\n')
    tmp=map(int, map(float, rr[500].split(' ')[:256]))
    print(list(tmp))
    print('\n\n\n')
    tmp=map(int, map(float, rr[10].split(' ')[:256]))
    print(list(tmp))
    print('\n\n\n')
    tmp=map(int, map(float, rr[3].split(' ')[:256]))
    print(list(tmp))