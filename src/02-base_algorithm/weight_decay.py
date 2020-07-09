from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import myutils


def init_params():
    # 生成一个随机的正态分布
    # scale：标准差
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]


def l2_penalty(w):
    return (w**2).sum() / 2


def fit_and_plot(lambd):
    w, b = init_params()
    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        # X：train_features，y：train_labels
        for X, y in train_iter:
            with autograd.record():
                # l是有关⼩批量X和y的损失，加上L2范数惩罚项
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            # 小批量损失对模型参数求梯度
            l.backward()
            # 小批量随机梯度下降求参数
            myutils.sgd([w, b], lr, batch_size)
        # 每次训练的损失
        train_loss.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_loss.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    # 这两行放在for循环内会导致l = loss(net(X, w, b), y) + lambd * l2_penalty(w)执行报错
    myutils.semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss',
                     range(1, num_epochs + 1), test_loss, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())


def fit_and_plot_gluon(wd):
    net_gluon = nn.Sequential()
    net_gluon.add(nn.Dense(1))
    net_gluon.initialize(init.Normal(sigma=1))
    trainer_w = gluon.Trainer(net_gluon.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd': wd})
    trainer_b = gluon.Trainer(net_gluon.collect_params('.*bias'), 'sgd', {'learning_rate': lr})
    train_loss, test_loss = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net_gluon(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_loss.append(loss(net_gluon(train_features), train_labels).mean().asscalar())
        test_loss.append(loss(net_gluon(test_features), test_labels).mean().asscalar())
    myutils.semilogy(range(1, num_epochs + 1), train_loss, 'epochs', 'loss',
                     range(1, num_epochs + 1), test_loss, ['train', 'test'])
    print('L2 norm of w:', net_gluon[0].weight.data().norm().asscalar())


# 训练集，测试集
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.5

# 数据集
features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
# 训练集20x200，测试集100x200
train_features, test_features = features[:n_train, :], features[n_train:, :]
# 生成数据
train_labels, test_labels = labels[:n_train], labels[n_train:]

batch_size, num_epochs, lr = 1, 100, 0.003
# 模型，损失函数
net, loss = myutils.linreg, myutils.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

# fit_and_plot(lambd=0)
# fit_and_plot(lambd=3)
fit_and_plot_gluon(wd=0)
fit_and_plot_gluon(wd=3)