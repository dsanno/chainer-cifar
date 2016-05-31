import numpy as np
import six
from scipy.misc import imresize, imrotate

from chainer import functions as F
from chainer import cuda
from chainer import Variable

class CifarTrainer(object):
    def __init__(self, net, optimizer, epoch_num=100, batch_size=100, device_id=-1):
        self.net = net
        self.optimizer = optimizer
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.device_id = device_id
        if device_id >= 0:
            self.xp = cuda.cupy
            self.net.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, x, y, test_x=None, test_y=None, callback=None):
        if self.device_id >= 0:
            with cuda.cupy.cuda.Device(self.device_id):
                return self.__fit(x, y, test_x, test_y, callback)
        else:
            return self.__fit(x, y, test_x, test_y, callback)

    def __fit(self, x, y, test_x, test_y, callback):
        batch_size = self.batch_size
        for epoch in six.moves.range(self.epoch_num):
            perm = np.random.permutation(len(x))
            train_loss = 0
            train_acc = 0
            for i in six.moves.range(0, len(x), self.batch_size):
                self.net.zerograds()
                batch_index = perm[i:i + batch_size]
                x_batch = self.__trans_image(x[batch_index])
                loss, acc = self.__forward(x_batch, y[batch_index])
                loss.backward()
                self.optimizer.update()
                train_loss += float(loss.data) * len(x_batch)
                train_acc += float(acc.data) * len(x_batch)
            train_loss /= len(x)
            train_acc /= len(x)
            test_loss = 0
            test_acc = 0
            if test_x is not None and test_y is not None:
                for i in six.moves.range(0, len(test_x), self.batch_size):
                    x_batch = self.__crop_image(test_x[i:i + batch_size])
                    loss, acc = self.__forward(x_batch, test_y[i:i + batch_size], train=False)
                    test_loss += float(loss.data) * len(x_batch)
                    test_acc += float(acc.data) * len(x_batch)
                test_loss /= len(test_x)
                test_acc /= len(test_x)
            if callback is not None:
                callback(epoch, self.net, self.optimizer, train_loss, train_acc, test_loss, test_acc)

    def __forward(self, batch_x, batch_t, train=True):
        xp = self.xp
        x = Variable(xp.asarray(batch_x), volatile=not train)
        t = Variable(xp.asarray(batch_t), volatile=not train)
        y = self.net(x, train=train)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        return loss, acc

    def __trans_image(self, x):
        size = 24
        n = x.shape[0]
        images = np.zeros((n, 3, size, size), dtype=np.float32)
        mirror = np.random.randint(2, size=n)
        scale = np.random.randint(3, size=n)
        rotate = np.random.randint(-2, 3, size=n)
        offset = np.random.randint(5, size=(n, 2))
        shift = np.random.uniform(-20, 20, n).astype(np.float32).reshape((n, 1, 1, 1))
        contrast = np.random.uniform(0.8, 1.2, n).astype(np.float32).reshape((n, 1, 1, 1))
        for i in six.moves.range(n):
            s = scale[i]
            top, left = offset[i]
            top *= s + 1
            left *= s + 1
            image = x[i].transpose((1, 2, 0))
            if s == 0:
                image = imresize(image, (28, 28))
            elif s == 2:
                image = imresize(image, (36, 36))
            if rotate[i] != 0:
                image = imrotate(image, rotate[i] * 4)
            image = image.transpose((2, 0, 1))
            if mirror[i] > 0:
                images[i,:,:,::-1] = image[:,top:top + size,left:left + size]
            else:
                images[i,:,:,:] = image[:,top:top + size,left:left + size]
        return images * contrast + shift

    def __crop_image(self, x):
        size = 24
        offset = (32 - size) // 2
        return x[:,:,offset:offset + size,offset:offset + size]
