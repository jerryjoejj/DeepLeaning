import numpy as np
import nn


def file2matrix(filename):
    fr = open(filename, encoding='utf-8')
    number_of_lines = len(fr.readlines())
    return_mat = np.zeros((number_of_lines, 27))  ##np.zeros就是创建规定行列数的矩阵
    class_label_vector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  ##删除原始数据中的空白符
        list_from_line = line.split()  ##数据分割，这里的数据是通过tab，所以用\t
        for num in list_from_line[0: 27]:
            if num == "?":
                return_mat[index, :] = 0
            else:
                return_mat[index, :] = num
            # return_mat[index, :] = list_from_line[0:27]  ##切割的数据只要前27列
        if list_from_line[-1] == 1:
            class_label_vector.append(1)
        elif list_from_line[-1] == 2:
            class_label_vector.append(2)
        index += 1
    return return_mat, class_label_vector  ##输出特征向量和特征值


dating_data_mat, dating_labels = file2matrix('horse-colic.data')
