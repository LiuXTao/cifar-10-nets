import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# 使用tensorflow时用到此文件
def one_hot(index):
    o = np.zeros(10)
    o[index] = 1.0
    return o

def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        i = 1
        # datadict is a dict type
        datadict = pickle.load(f, encoding='latin1')
        x = datadict['data']
        y = datadict['labels']
        x = np.array(x)
        # if i==1:
        #     new = x.reshape(10000, 3, 32, 32)
        #     red = new[99][0].reshape(1024, 1)
        #     green = new[99][1].reshape(1024, 1)
        #     blue = new[99][2].reshape(1024, 1)
        #     pic = np.hstack((red, green, blue)).reshape(32, 32, 3)
        #     plt.imshow(pic)
        #     plt.legend()
        #     plt.show()
        #     i += 1
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)
        y = np.array(y)
        y_onehot = []
        for i in y:
            y_onehot.append(one_hot(i))
        y_onehot = np.array(y_onehot)
    return x, y_onehot

'''
# 读取数据

# 返回格式：x_train, y_train, x_test, y_test
'''
def load_cifar10(dir):
    xlist = []
    ylist = []
    for b in range(1, 6):
        file = os.path.join(dir, 'data_batch_%d' % b)
        x, y = load_cifar10_batch(file)
        xlist.append(x)
        ylist.append(y)
    x = np.concatenate(xlist)
    y = np.concatenate(ylist)
    test_dir = os.path.join(dir, 'test_batch')
    x_test, y_test = load_cifar10_batch(test_dir)
    return x, y, x_test, y_test


# if __name__ == '__main__':
#     x_train, y_train, x_test, y_test = load_cifar10('./cifar-10-batches-py/')
#     print(len(x_train))
#     print(len(y_train))
#     print(x_train, y_train)

'''
数组拼接方法：
1. alist.extend(blist)  alist = np.array(alist)

2. np.append(a, 10)

3.  np.concatenate((a,b,c),axis=0)  # 默认情况下，axis=0可以不写, a,b,c 类型为array, 但是np.concatenate()的参数为列表或者元组类型

查看文件夹内有多少文件: ls|wc -l
'''