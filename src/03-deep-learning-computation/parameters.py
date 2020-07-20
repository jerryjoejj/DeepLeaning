from mxnet import init, nd
from mxnet.gluon import nn


class MyInit(init.Initializer):
    """
    自定义初始化方法
    """
    def _init_weight(self, name, arr):
        print('Init', name, arr.shape)
        arr[:] = nd.random.uniform(low=-10, high=10, shape=arr.shape)
        arr *= arr.abs() >= 5


# net = nn.Sequential()
# net.add(nn.Dense(256, activation='relu'))
# net.add(nn.Dense(10))
# net.initialize()
net = nn.Sequential()
# 共享模型参数
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'), shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
# 前向计算
y = net(x)
print(y)
print(net[1].weight.data()[0] == net[2].weight.data()[0])
# 索引0表示隐藏层为Sequential实例最先添加的层
# print(net[0].params)
# print(net[0].params['dense0_weight'])
# print(net[0].weight)
# 获取模型参数值
# print(net[0].weight.data())
# print(net[1].weight.data())
# print(net[0].weight.grad())
# print(net.collect_params())
# net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
# print(net[0].weight.data()[0])
# 使用常数初始化权重参数
# net.initialize(init=init.Constant(1), force_reinit=True)
# print(net[0].weight.data()[0])
# 对特定参数进行初始化
# net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
# print(net[0].weight.data()[0])

net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[0])
