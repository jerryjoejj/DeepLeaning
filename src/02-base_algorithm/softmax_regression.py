import d2lzh as d2l
from mxnet import autograd, nd
from matplotlib import pyplot as plt
from myutils.util import show_fashion_mnist2


def softmax(x):
    """
    softmax函数
    :param x:
    :return:
    """
    x_exp = x.exp()
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition


# X = nd.random.normal(shape=(2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(axis=1))

def net(x, num_inputs, W, b):
    """
    定义模型
    :param num_inputs:
    :param x:
    :return:
    """
    return softmax(nd.dot(x.reshape((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    """
    定义损失函数
    :param y_hat:
    :param y:
    :return:
    """
    return -nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    """
    计算分类准确率
    :param y_hat:
    :param y:
    :return:
    """
    # y_hat.argmax(axis=1) 返回矩阵y_hat每一行最大元素的索引
    # mean() 求均值
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net, num_inputs, W, b):
    """
    评价模型net在数据集data_iter上的准确率
    :param data_iter:
    :param net:
    :param num_inputs:
    :param W:
    :param b:
    :return:
    """
    acc_sum, n = 0.0, 0.0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X, num_inputs, W, b).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              num_inputs, W, b,
              params=None, lr=None, trainer=None):
    """
    开始训练
    :param net: 模型函数
    :param train_iter: 训练数据
    :param test_iter: 测试数据
    :param loss: 损失函数
    :param num_epochs: 迭代周期数
    :param batch_size:
    :param num_inputs:
    :param W:
    :param b:
    :param params:
    :param lr: 学习率
    :param trainer:
    :return:
    """
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X, num_inputs, W, b)
                data_loss = loss(y_hat, y).sum()
                data_loss.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_loss_sum += data_loss.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, num_inputs, W, b)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))


def main():
    batch_size = 256
    # 下载、读取数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
    b = nd.zeros(num_outputs)

    # 附上梯度，开辟梯度缓冲区
    W.attach_grad()
    b.attach_grad()

    num_epochs, lr = 5, 0.1
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
              num_inputs, W, b,
              [W, b], lr)
    for X, y in test_iter:
        print(X, y)
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X, num_inputs, W, b).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    show_fashion_mnist2(X[0:9], titles[0:9])


if __name__ == '__main__':
    main()
