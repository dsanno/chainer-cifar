import math
import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

from functions import arrange
from links import Hadamard

class BatchConv2D(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0, activation=F.relu):
        super(BatchConv2D, self).__init__(
            conv=L.Convolution2D(ch_in, ch_out, ksize, stride, pad),
            bn=L.BatchNormalization(ch_out),
        )
        self.activation=activation

    def __call__(self, x, train):
        h = self.bn(self.conv(x), test=not train)
        if self.activation is None:
            return h
        return F.relu(h)

class FastFood(chainer.Chain):
    def __init__(self, size):
        wscale = math.sqrt(2)
        super(FastFood, self).__init__(
            diag1=L.Linear(size, 1, nobias=True),
            diag2=L.Linear(size, 1, nobias=True),
            diag3=L.Linear(size, 1, nobias=True),
            hadamard=Hadamard(size),
        )

        self.add_persistent('pi', np.random.permutation(size))

    def __call__(self, x):
        shape = x.data.shape
        if len(shape) != 2:
            h = F.reshape(x, (shape[0], -1))
        else:
            h = x
        shape = h.data.shape
        h = F.broadcast_to(self.diag1.W, shape) * h
        h = self.hadamard(h)
        h = arrange(h, chainer.Variable(self.pi, volatile=h.volatile))
        h = F.broadcast_to(self.diag2.W, shape) * h
        h = self.hadamard(h)
        h = F.broadcast_to(self.diag3.W, shape) * h
        return h

class ResidualBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, stride=1, swapout=False, skip_ratio=0, activation1=F.relu, activation2=F.relu):
        w = math.sqrt(2)
        super(ResidualBlock, self).__init__(
            conv1=L.Convolution2D(ch_in, ch_out, 3, stride, 1, w),
            bn1=L.BatchNormalization(ch_out),
            conv2=L.Convolution2D(ch_out, ch_out, 3, 1, 1, w),
            bn2=L.BatchNormalization(ch_out),
        )
        self.activation1 = activation1
        self.activation2 = activation2
        self.skip_ratio = skip_ratio
        self.swapout = swapout

    def __call__(self, x, train):
        skip = False
        if train and self.skip_ratio > 0 and np.random.rand() < self.skip_ratio:
            skip = True
        sh, sw = self.conv1.stride
        c_out, c_in, kh, kw = self.conv1.W.data.shape
        b, c, hh, ww = x.data.shape
        if sh == 1 and sw == 1:
            shape_out = (b, c_out, hh, ww)
        else:
            hh = (hh + 2 - kh) // sh + 1
            ww = (ww + 2 - kw) // sw + 1
            shape_out = (b, c_out, hh, ww)
        h = x
        if x.data.shape != shape_out:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = shape_out[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=not train)
            x = F.concat((p, x))
            if x.data.shape[2:] != shape_out[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        if skip:
            return x
        h = self.bn1(self.conv1(h), test=not train)
        if self.activation1 is not None:
            h = self.activation1(h)
        h = self.bn2(self.conv2(h), test=not train)
        if not train:
            h = h * (1 - self.skip_ratio)
        if self.swapout:
            h = F.dropout(h, train=train) + F.dropout(x, train=train)
        else:
            h = h + x
        if self.activation2 is not None:
            return self.activation2(h)
        else:
            return h

class IdentityMappingBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, stride=1, swapout=False, activation1=F.relu, activation2=F.relu):
        w = math.sqrt(2)
        super(IdentityMappingBlock, self).__init__(
            bn1=L.BatchNormalization(ch_in),
            conv1=L.Convolution2D(ch_in, ch_out, 3, stride, 1, w),
            conv2=L.Convolution2D(ch_out, ch_out, 3, 1, 1, w),
            bn2=L.BatchNormalization(ch_out),
        )
        self.activation1 = activation1
        self.activation2 = activation2
        self.swapout = swapout

    def __call__(self, x, train):
        h = self.bn1(x, test=not train)
        if self.activation1 is not None:
            h = self.activation1(h)
        h = self.conv1(h)
        h = self.bn2(h, test=not train)
        if self.activation2 is not None:
            h = self.activation2(h)
        h = self.conv2(h)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=not train)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        if self.swapout:
            return F.dropout(h, train=train) + F.dropout(x, train=train)
        else:
            return h + x

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 64, 5, stride=1, pad=2),
            conv2=L.Convolution2D(64, 64, 5, stride=1, pad=2),
            conv3=L.Convolution2D(64, 128, 5, stride=1,
            pad=2),
            l1=L.Linear(4 * 4 * 128, 1000),
            l2=L.Linear(1000, 10),
        )

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3, 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 3, 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 3, 2)
        h4 = F.relu(self.l1(F.dropout(h3, train=train)))
        return self.l2(F.dropout(h4, train=train))

class CNNBN(chainer.Chain):
    def __init__(self):
        super(CNNBN, self).__init__(
            bconv1=BatchConv2D(3, 64, 5, stride=1, pad=2),
            bconv2=BatchConv2D(64, 64, 5, stride=1, pad=2),
            bconv3=BatchConv2D(64, 128, 5, stride=1, pad=2),
            l1=L.Linear(4 * 4 * 128, 1000),
            l2=L.Linear(1000, 10),
        )

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(self.bconv1(x, train), 3, 2)
        h2 = F.max_pooling_2d(self.bconv2(h1, train), 3, 2)
        h3 = F.max_pooling_2d(self.bconv3(h2, train), 3, 2)
        h4 = F.relu(self.l1(F.dropout(h3, train=train)))
        return self.l2(F.dropout(h4, train=train))

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            bconv1_1=BatchConv2D(3, 64, 3, stride=1, pad=1),
            bconv1_2=BatchConv2D(64, 64, 3, stride=1, pad=1),
            bconv2_1=BatchConv2D(64, 128, 3, stride=1, pad=1),
            bconv2_2=BatchConv2D(128, 128, 3, stride=1, pad=1),
            bconv3_1=BatchConv2D(128, 256, 3, stride=1, pad=1),
            bconv3_2=BatchConv2D(256, 256, 3, stride=1, pad=1),
            bconv3_3=BatchConv2D(256, 256, 3, stride=1, pad=1),
            bconv3_4=BatchConv2D(256, 256, 3, stride=1, pad=1),
            fc4=L.Linear(4 * 4 * 256, 1024),
            fc5=L.Linear(1024, 1024),
            fc6=L.Linear(1024, 10),
        )

    def __call__(self, x, train=True):
        h = self.bconv1_1(x, train)
        h = self.bconv1_2(h, train)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25, train=train)
        h = self.bconv2_1(h, train)
        h = self.bconv2_2(h, train)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25, train=train)
        h = self.bconv3_1(h, train)
        h = self.bconv3_2(h, train)
        h = self.bconv3_3(h, train)
        h = self.bconv3_4(h, train)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25, train=train)
        h = F.relu(self.fc4(F.dropout(h, train=train)))
        h = F.relu(self.fc5(F.dropout(h, train=train)))
        h = self.fc6(h)
        return h

class VGGFastFood(chainer.Chain):
    def __init__(self):
        super(VGGFastFood, self).__init__(
            bconv1_1=BatchConv2D(3, 64, 3, stride=1, pad=1),
            bconv1_2=BatchConv2D(64, 64, 3, stride=1, pad=1),
            bconv2_1=BatchConv2D(64, 128, 3, stride=1, pad=1),
            bconv2_2=BatchConv2D(128, 128, 3, stride=1, pad=1),
            bconv3_1=BatchConv2D(128, 256, 3, stride=1, pad=1),
            bconv3_2=BatchConv2D(256, 256, 3, stride=1, pad=1),
            bconv3_3=BatchConv2D(256, 256, 3, stride=1, pad=1),
            bconv3_4=BatchConv2D(256, 256, 3, stride=1, pad=1),
            ff4=FastFood(4096),
            ff5=FastFood(4096),
            fc6=L.Linear(4096, 10),
        )

    def __call__(self, x, train=True):
        h = self.bconv1_1(x, train)
        h = self.bconv1_2(h, train)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25, train=train)
        h = self.bconv2_1(h, train)
        h = self.bconv2_2(h, train)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25, train=train)
        h = self.bconv3_1(h, train)
        h = self.bconv3_2(h, train)
        h = self.bconv3_3(h, train)
        h = self.bconv3_4(h, train)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25, train=train)
        h = F.relu(self.ff4(F.dropout(h, train=train)))
        h = F.relu(self.ff5(F.dropout(h, train=train)))
        h = self.fc6(h)
        return h

class ResidualNet(chainer.Chain):
    def __init__(self, depth=18, swapout=False, skip=False):
        super(ResidualNet, self).__init__()
        links = [('bconv1', BatchConv2D(3, 16, 3, 1, 1), True)]
        skip_size = depth * 3 - 3
        for i in six.moves.range(depth):
            if skip:
                skip_ratio = float(i) / skip_size * 0.5
            else:
                skip_ratio = 0
            links.append(('res{}'.format(len(links)), ResidualBlock(16, 16, swapout=swapout, skip_ratio=skip_ratio, ), True))
        links.append(('res{}'.format(len(links)), ResidualBlock(16, 32, stride=2, swapout=swapout), True))
        for i in six.moves.range(depth - 1):
            if skip:
                skip_ratio = float(i + depth) / skip_size * 0.5
            else:
                skip_ratio = 0
            links.append(('res{}'.format(len(links)), ResidualBlock(32, 32, swapout=swapout, skip_ratio=skip_ratio), True))
        links.append(('res{}'.format(len(links)), ResidualBlock(32, 64, stride=2, swapout=swapout), True))
        for i in six.moves.range(depth - 1):
            if skip:
                skip_ratio = float(i + depth * 2 - 1) / skip_size * 0.5
            else:
                skip_ratio = 0
            links.append(('res{}'.format(len(links)), ResidualBlock(64, 64, swapout=swapout, skip_ratio=skip_ratio), True))
        links.append(('_apool{}'.format(len(links)), F.AveragePooling2D(8, 1, 0, False, True), False))
        links.append(('fc{}'.format(len(links)), L.Linear(64, 10), False))

        for name, f, _with_train in links:
            if not name.startswith('_'):
                self.add_link(*(name, f))
        self.layers = links

    def __call__(self, x, train=True):
        h = x
        for name, f, with_train in self.layers:
            if with_train:
                h = f(h, train=train)
            else:
                h = f(h)
        return h

class IdentityMapping(chainer.Chain):
    def __init__(self, depth=18, swapout=False):
        super(IdentityMapping, self).__init__()
        links = [('bconv1', BatchConv2D(3, 16, 3, 1, 1), True)]
        for i in six.moves.range(depth):
            links.append(('res{}'.format(len(links)), IdentityMappingBlock(16, 16, swapout=swapout, activation1=F.relu, activation2=F.relu), True))
        links.append(('res{}'.format(len(links)), IdentityMappingBlock(16, 32, stride=2, swapout=swapout, activation1=F.relu, activation2=F.relu), True))
        for i in six.moves.range(depth - 1):
            links.append(('res{}'.format(len(links)), IdentityMappingBlock(32, 32, swapout=swapout, activation1=F.relu, activation2=F.relu), True))
        links.append(('res{}'.format(len(links)), IdentityMappingBlock(32, 64, stride=2, swapout=swapout, activation1=F.relu, activation2=F.relu), True))
        for i in six.moves.range(depth - 1):
            links.append(('res{}'.format(len(links)), IdentityMappingBlock(64, 64, swapout=swapout, activation1=F.relu, activation2=F.relu), True))
        links.append(('_apool{}'.format(len(links)), F.AveragePooling2D(8, 1, 0, False, True), False))
        links.append(('fc{}'.format(len(links)), L.Linear(64, 10), False))

        for name, f, _with_train in links:
            if not name.startswith('_'):
                self.add_link(*(name, f))
        self.layers = links

    def __call__(self, x, train=True):
        h = x
        for name, f, with_train in self.layers:
            if with_train:
                h = f(h, train=train)
            else:
                h = f(h)
        return h
