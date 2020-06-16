from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon


def main():
    # 生成数据集
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)

    # 读取数据集
    batch_size = 10
    data_set = gdata.ArrayDataset(features, labels)
    data_iter = gdata.DataLoader(data_set, batch_size, shuffle=True)

    for X, y in data_iter:
        print(X, y)
        break

    net = nn.Sequential()
    net.add(nn.Dense(1))
    # 初始化模型参数
    net.initialize(init.Normal(sigma=0.01))
    # 定义损失函数
    loss = gloss.L2Loss()
    # 定义优化算法
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            with autograd.record():
                data_loss = loss(net(X), y)
                # 自动求梯度
                data_loss.backward()
            trainer.step(batch_size)
        data_loss = loss(net(features), labels)
        print('epoch %d, loss: %f' % (epoch, data_loss.mean().asnumpy()))

    dense = net[0]
    print(true_w, dense.weight.data())
    print(true_b, dense.bias.data())


if __name__ == '__main__':
    main()