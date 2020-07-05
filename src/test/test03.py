import numpy as np
import matplotlib.pyplot as plt


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 4))  ##np.zeros就是创建规定行列数的矩阵
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  ##删除原始数据中的空白符
        listFromLine = line.split('\t')  ##数据分割，这里的数据是通过tab，所以用\t
        returnMat[index, :] = listFromLine[0:4]  ##切割的数据只要前4列
        if listFromLine[-1] == 'setosa':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'versicolor':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'virginica':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector  ##输出特征向量和特征值


##0-1标准化
def autoNorm(dataSet):  ##特征数据
    minVals = dataSet.min(0)  ##0代表求数据集一列中的最小值
    maxVals = dataSet.max(0)  ##0代表求数据集一列中的最大值
    normDataSet = np.zeros(dataSet.shape)  ##shape返回行列数
    normDataSet = (dataSet - minVals) / (maxVals - minVals)
    return normDataSet


datingDataMat, datingLabels = file2matrix('iris.txt')

normalSet = autoNorm(datingDataMat)
print(normalSet)
print(datingLabels)
plt.scatter(normalSet[0:], normalSet[1:])
plt.show()
