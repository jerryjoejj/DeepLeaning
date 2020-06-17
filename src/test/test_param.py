from mxnet import autograd, nd


def fun(x,y):
    x[0] = 10
    y += 1


def main():
    # 可变对象
    x = [1]
    # 不可变对象
    y = 2
    fun(x, y)
    print(x, y)


if __name__ == '__main__':
    main()