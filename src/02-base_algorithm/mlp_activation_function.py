from mxnet import autograd, nd
import d2lzh as d2l


def xygplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()


x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()

xygplot(x, y, 'relu')
# TODO 必须执行y.backward()，否则图形错误
# 求梯度
y.backward()
xygplot(x, x.grad, 'grad of relu')

with autograd.record():
    y = x.sigmoid()

xygplot(x, y, "sigmoid")

y.backward()
xygplot(x, x.grad, 'grad of sigmoid')

with autograd.record():
    y = x.tanh()

xygplot(x, y, 'tanh')

y.backward()
xygplot(x, x.grad, 'grad of tanh')