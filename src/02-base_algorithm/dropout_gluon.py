from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import myutils

drop_prob1, drop_prob2 = 0.2, 0.5
num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = myutils.load_data_fashion_mnist(batch_size)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob1), # 在第一个全连接层后添加丢弃层
        nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob2), # 在第二个全连接层后添加丢弃层
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
myutils.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)