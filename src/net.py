import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L

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

class ResidualBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, stride=1, ksize=1, activation1=F.relu, activation2=F.relu):
        w = math.sqrt(2)
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(ch_in, ch_out, 3, stride, 1, w),
            bn1=L.BatchNormalization(ch_out),
            conv2=L.Convolution2D(ch_out, ch_out, 3, 1, 1, w),
            bn2=L.BatchNormalization(ch_out),
        )
        self.activation1 = activation1
        self.activation2 = activation2

    def __call__(self, x, train):
        h = self.bn1(self.conv1(x), test=not train)
        if self.activation1 is not None:
            h = self.activation1(h)
        h = self.bn2(self.conv2(h), test=not train)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=not train)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        if self.activation2 is not None:
            return self.activation2(h + x)
        else:
            return h + x

class Conv(chainer.Chain):
    def __init__(self):
        super(Conv, self).__init__(
            conv1=L.Convolution2D(3, 64, 5, stride=1, pad=2),
            conv2=L.Convolution2D(64, 64, 5, stride=1, pad=2),
            conv3=L.Convolution2D(64, 128, 5, stride=1,
            pad=2),
            l1=L.Linear(3 * 3 * 128, 1000),
            l2=L.Linear(1000, 10),
        )

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3, 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 3, 2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), 3, 2)
        h4 = F.relu(self.l1(F.dropout(h3, train=train)))
        return self.l2(F.dropout(h4, train=train))

class ConvWithBatch(chainer.Chain):
    def __init__(self):
        super(ConvWithBatch, self).__init__(
            bconv1=BatchConv2D(3, 64, 5, stride=1, pad=2),
            bconv2=BatchConv2D(64, 64, 5, stride=1, pad=2),
            bconv3=BatchConv2D(64, 128, 5, stride=1, pad=2),
            l1=L.Linear(3 * 3 * 128, 1000),
            l2=L.Linear(1000, 10),
        )

    def __call__(self, x, train=True):
        h1 = F.max_pooling_2d(self.bconv1(x, train), 3, 2)
        h2 = F.max_pooling_2d(self.bconv2(h1, train), 3, 2)
        h3 = F.max_pooling_2d(self.bconv3(h2, train), 3, 2)
        h4 = F.relu(self.l1(F.dropout(h3, train=train)))
        return self.l2(F.dropout(h4, train=train))

class VGGLike(chainer.Chain):
    def __init__(self):
        super(VGGLike, self).__init__(
            bconv1=BatchConv2D(3, 32, 3, stride=1, pad=1),
            bconv2=BatchConv2D(32, 32, 3, stride=1, pad=1),
            bconv3=BatchConv2D(32, 64, 3, stride=1, pad=1),
            bconv4=BatchConv2D(64, 64, 3, stride=1, pad=1),
            bconv5=BatchConv2D(64, 128, 3, stride=1, pad=1),
            bconv6=BatchConv2D(128, 128, 3, stride=1, pad=1),
            l1=L.Linear(7 * 7 * 128, 1024),
            l2=L.Linear(1024, 10),
        )

    def __call__(self, x, train=True):
        h1 = self.bconv1(x, train)
        h2 = F.max_pooling_2d(self.bconv2(h1, train), 2)
        h3 = self.bconv3(h2, train)
        h4 = F.max_pooling_2d(self.bconv4(h3, train), 2)
        h5 = self.bconv5(h4, train)
        h6 = self.bconv6(h5, train)
        h7 = F.relu(self.l1(F.dropout(h6, train=train)))
        h8 = self.l2(F.dropout(h7, train=train))
        return h8
