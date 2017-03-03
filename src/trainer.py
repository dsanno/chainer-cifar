import numpy as np
import six
from scipy.misc import imresize, imrotate
import time

import chainer
from chainer.dataset import convert
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

    def fit(self, train_data, valid_data=None, test_data=None, callback=None):
        if self.device_id >= 0:
            with cuda.cupy.cuda.Device(self.device_id):
                return self.__fit(train_data, valid_data, test_data, callback)
        else:
            return self.__fit(train_data, valid_data, test_data, callback)

    def __fit(self, train_data, valid_data, test_data, callback):
        batch_size = self.batch_size
        train_iterator = chainer.iterators.SerialIterator(train_data, self.batch_size, repeat=True, shuffle=True)
        train_loss = 0
        train_acc = 0
        num = 0
        while train_iterator.epoch < self.epoch_num:
            batch = train_iterator.next()
            x_batch, y_batch = convert.concat_examples(batch, self.device_id)
            loss, acc = self.__forward(x_batch, y_batch)
            self.net.cleargrads()
            loss.backward()
            self.optimizer.update()
            train_loss += float(loss.data) * len(x_batch)
            train_acc += float(acc.data) * len(x_batch)
            num += len(x_batch)
            if not train_iterator.is_new_epoch:
                continue
            train_loss /= num
            train_acc /= num
            valid_loss = 0
            valid_acc = 0
            if valid_data is not None:
                valid_loss, valid_acc = self.__evaluate(valid_data)
            test_loss = 0
            test_acc = 0
            test_time = 0
            if test_data is not None:
                start_clock = time.clock()
                test_loss, test_acc = self.__evaluate(test_data)
                test_time = time.clock() - start_clock
            if callback is not None:
                callback(train_iterator.epoch, self.net, self.optimizer, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, test_time)
            train_loss = 0
            train_acc = 0
            num = 0
        train_iterator.finalize()

    def __evaluate(self, data):
        iterator = chainer.iterators.SerialIterator(data, self.batch_size, repeat=False, shuffle=False)
        total_loss = 0
        total_acc = 0
        num = 0
        for batch in iterator:
            x_batch, y_batch = convert.concat_examples(batch, self.device_id)
            loss, acc = self.__forward(x_batch, y_batch, train=False)
            total_loss += float(loss.data) * len(x_batch)
            total_acc += float(acc.data) * len(x_batch)
            num += len(x_batch)
        iterator.finalize()
        return total_loss / num, total_acc / num

    def __forward(self, batch_x, batch_t, train=True):
        xp = self.xp
        x = Variable(xp.asarray(batch_x), volatile=not train)
        t = Variable(xp.asarray(batch_t), volatile=not train)
        y = self.net(x, train=train)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        return loss, acc
