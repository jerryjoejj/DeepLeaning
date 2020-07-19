import myutils
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, data as gdata, nn
import numpy as np
import pandas as pd


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net


def log_rmse(net, features, labels):
    """
    使用对数均方误差评价模型
    :param net:
    :param features:
    :param labels:
    :return:
    """
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs,
          learning_rate, weight_decay, batch_size):
    """
    模型训练
    :param net:             模型
    :param train_features:  训练集
    :param train_labels:    训练标签
    :param test_features:   测试集
    :param test_labels:     测试标签
    :param num_epochs:      迭代次数
    :param learning_rate:   学习率
    :param weight_decay:    权重衰减超参数
    :param batch_size:      每次训练数据集大小
    :return:
    """
    # 训练集损失，测试集损失
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 使用adam优化算法
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, x, y):
    """
    第i折交叉验证时所需要的训练和验证数据
    :param k:
    :param i:
    :param x:
    :param y:
    :return:
    """
    assert k > 1
    fold_size = x.shape[0] // k
    x_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = nd.concat(x_train, x_part ,dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return x_train, y_train, x_valid, y_valid


def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """
    k折交叉验证，训练k次并返回训练和验证的平均误差
    :param k:
    :param x_train:
    :param y_train:
    :param num_epochs:
    :param learning_rate:
    :param weight_decay:
    :param batch_size:
    :return: 返回平均训练损失和平均验证损失
    """
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net()
        # train_ls: 训练损失，valid_ls：验证损失
        # *data：train_features, train_labels, test_features, test_labels
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            myutils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                             range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('fold %d train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_deacy, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_deacy, batch_size)
    myutils.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


# 读取数据
train_data = pd.read_csv('data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('data/kaggle_house_pred_test.csv')

# 连接训练集和测试集
all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))

# 预处理数据集
# 取表头，且列对象不是object
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# pandas.DataFrame.apply：对DataFrame里的所有值执行方法中的计算
# mean方法：矩阵平均值
# std方法：矩阵标准差
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# pandas.DataFrame.fillna：将NA/NAN替换成特定的值或者特殊的方法
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# pandas.get_dummies：将类别变量转换成新增的虚拟变量
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# 转换为NDArray对象
n_train = train_data.shape[0]
# 训练集
train_features = nd.array(all_features[:n_train].values)
# 测试集
test_features = nd.array(all_features[n_train:].values)
# 训练标签
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

# 定义训练模型
loss = gloss.L2Loss()

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)