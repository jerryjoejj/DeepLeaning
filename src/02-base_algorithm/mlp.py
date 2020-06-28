import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss
import myutils as myutils


def relu(x):
    """
    定义激活函数
    :param x:
    :return:
    """
    return nd.maximum(x, 0)


def net(x):
    """
    定义模型
    :param x:
    :param num_inputs:
    :param params:
    :return:
    """
    x = x.reshape((-1, num_inputs))
    h = relu(nd.dot(x, params[0]) + params[1])
    return nd.dot(h, params[2]) + params[3]


batch_size = 256
# 下载、读取数据集
train_iter, test_iter = myutils.load_data_fashion_mnist(batch_size)
# 输入个数，输出个数，超参数隐藏单元
# 超参数隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
# 附上梯度
params = [W1, b1, W2, b2]
for param in params:
    param.attach_grad()

loss = gloss.SoftmaxCrossEntropyLoss()
num_epochs, lr = 5, 0.01
myutils.train_ch3(net, train_iter, test_iter, loss, num_epochs, num_inputs, batch_size, params, lr)
