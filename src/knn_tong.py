import numpy as np
import random
from sklearn import model_selection, neighbors
import pandas as pd


filepath = "D:/cancer.csv"
dataset = pd.read_csv(filepath, index_col=0, encoding="utf-8")


# print(dataset2)
# 分离data和label##
col = dataset.columns.values.tolist()
data = np.array(dataset[col])
random.shuffle(data)
dataset.replace('?', np.nan, inplace=True)  # -99999
dataset.dropna(inplace=True)  # 去掉无效数据
# print(df.shape)
# dataset.drop(['id'], 1, inplace=True)
# 将dataframe对象转换为NDArray
dataset2 = dataset.values
# print(dataset2)
X = dataset2[:, 2]
y = dataset2[:, 5]
# print('X:', X)
# print('y:', y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# print('X_train:', X_train)
# print('y_train:', y_train)
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train.reshape(1, -1).astype('int'), y_train.reshape(1, -1).astype('int'))
