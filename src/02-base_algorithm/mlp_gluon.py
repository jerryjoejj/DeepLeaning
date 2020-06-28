from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import myutils

# 定义模型
net = nn.Sequential()
# 定义隐藏全连接层，隐藏单元为256，输出单元为10，使用ReLu函数作为激活函数
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

batch_size = 256
# 下载、读取数据集
train_iter, test_iter = myutils.load_data_fashion_mnist(batch_size)
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
num_epochs = 5
# 训练模型
myutils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
