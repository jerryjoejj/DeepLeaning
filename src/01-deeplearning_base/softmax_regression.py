import d2lzh as d2l
from mxnet import autograd, nd


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
    return (y_hat.argmax(axis=1)) == y.astype('float32').mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0.0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              num_inputs, W, b,
              params=None, lr=None, trainer=None):
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
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().acccalar()
            n += y.size


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
