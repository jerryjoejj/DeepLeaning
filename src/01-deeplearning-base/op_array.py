from mxnet import nd
x = nd.arange(12)
#
# print(x)
# print(x.shape)
X = x.reshape((3, 4))
# print(X)
#
# a = nd.zeros((2, 3, 4))
# print(a)
#
# b = nd.ones((2, 3, 4))
# print(b)

Y = nd.array([[2, 1, 4, 3],
              [1, 2, 3, 4],
              [4, 3, 2, 1]])
print(Y)
# before = id(Y)
# Y = Y + X
# print(before)
# print(id(Y) == before)

# Z = nd.random.normal(0, 1, shape=(3, 4))
# print(Z)

# print(X + Y)
#
# print(X * Y)
# print(X / Y)
# print(Y.exp())
# print(nd.dot(X, Y.T))


# print(nd.concat(X, Y, dim=0))
# print(nd.concat(X, Y, dim=1))
# print(X == Y)
# print(X.sum())
# print(X.norm().asscalar())

# A = nd.arange(3).reshape((3, 1))
# B = nd.arange(2).reshape((1, 2))
#
# print(A)
# print(B)
# print(A + B)

# print(X[1:3])
# print(X)
# X[1, 2] = 2
# X[1:2, :] = 12
# print(X)

Z = Y.zeros_like()
before = id(Z)
# 开辟临时内存存储X，Y
# Z[:] = X + Y
nd.elemwise_add(X, Y, out=Z)
print(id(Z) == before)
