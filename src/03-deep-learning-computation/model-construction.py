from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    """
    无需定义反向传播函数，系统将通过自动求梯度而自动生成反向传播的backward函数
    """
    # 声明带有模型参数的层
    # 声明两个全连接层
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu') # 隐藏层
        self.output = nn.Dense(10) # 输出层

    def forward(self, *args):
        """
        定义模型的向前计算，即如何根据输入x计算返回所需要的模型输出
        :param args:
        :return:
        """
        return self.output(self.hidden(*args))


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, *args):
        for block in self._children.values():
            x = block(*args)
        return x


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(*kwargs)
        # 使用get_constent创建的随机权重参数不会在训练中迭代
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, *args):
        x = self.dense(*args)
        # 使用创建的常数参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层，等价两个全连接层共享参数
        x = self.dense(x)
        # 控制流
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, *args):
        return self.dense(self.net(x))


x = nd.random.uniform(shape=(2, 20))
# net = MLP()
# net.initialize()
# print(net(x))

# net = MySequential()
# net.add(nn.Dense(256, activation='relu'))
# net.add(nn.Dense(10))
# net.initialize()
# print(net(x))

# net = FancyMLP()
# net.initialize()
# print(net(x))
net = nn.Sequential()
# 嵌套调用
net.add(NestMLP(), nn.Dense(20), FancyMLP())
net.initialize()
print(net(x))