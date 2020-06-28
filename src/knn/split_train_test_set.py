# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:04:29 2020

@author: manai
"""
from myutils.read_data_set import file2matrix
from myutils.data_normalization import auto_norm
from myutils.knn import knn


def main():
    file_name = "iris.txt"
    data_set, dating_labels = file2matrix(file_name)
    normal_set = auto_norm(data_set)
    # 改变m和k值可能可以调整错误率
    m = 0.8
    data_size = normal_set.shape[0]
    print('数据集总行数：', data_size)
    train_size = int(m * data_size)
    test_size = int((1 - m) * data_size)

    k = 5
    results = []
    error = 0
    for i in range(test_size):
        results = knn(normal_set[train_size + i - 1, :], normal_set[0:train_size, ], dating_labels[0:train_size], k)
        if results != dating_labels[train_size + i]:
            error += 1
    print('错误率：', error / test_size)


if __name__ == '__main__':
    main()
