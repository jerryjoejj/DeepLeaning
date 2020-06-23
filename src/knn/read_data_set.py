# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:34:21 2020

@author: manai
"""

import numpy as np


def file2matrix(filename):
    fr = open(filename)
    number_of_lines = len(fr.readlines())
    # np.zeros就是创建规定行列数的矩阵
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 删除原始数据中的空白符
        line = line.strip()
        # 数据分割，这里的数据是通过tab，所以用\t
        list_from_line = line.split('\t')
        # 切割的数据只要前三列
        return_mat[index, :] = list_from_line[0:3]
        if list_from_line[-1] == 'setosa':
            class_label_vector.append(1)
        elif list_from_line[-1] == 'versicolor':
            class_label_vector.append(2)
        elif list_from_line[-1] == 'virginica':
            class_label_vector.append(3)
        index += 1
    # 输出特征向量和特征值
    return return_mat, class_label_vector
