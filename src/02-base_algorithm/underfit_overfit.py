from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import myutils


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    myutils.set_figsize(figsize)
    myutils.plt.xlabel(x_label)
    myutils.plt.ylabel(y_label)
    myutils.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        myutils.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        myutils.plt.legend(legend)
    myutils.plt.show()


n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2]
          + true_b)
labels += nd.random.normal(scale=0.01, shape=labels.shape)



