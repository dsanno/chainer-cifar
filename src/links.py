import numpy
import six

import chainer
from chainer import link

import functions

class Hadamard(link.Link):

    """Hadamard transform."""

    def __init__(self, size):
        super(Hadamard, self).__init__()
        bit_len = 0
        j = size
        while j > 0:
            bit_len += 1
            j //= 2
        t = numpy.arange(size, dtype=numpy.int32)
        transpose = numpy.zeros((bit_len, size), dtype=numpy.int32)
        sign = numpy.zeros((bit_len, size), dtype=numpy.float32)
        i = 0
        j = size
        while j > 1:
            next_size = j // 2
            index1 = t % j < next_size
            index2 = numpy.logical_not(index1)
            transpose[i, index1] = t[index2]
            transpose[i, index2] = t[index1]
            sign[i, index1] = 1
            sign[i, index2] = -1
            j = next_size
            i += 1
        self.add_persistent('transpose', transpose)
        self.add_persistent('sign', sign)

    def __call__(self, x):
        t = chainer.Variable(self.transpose, volatile=x.volatile)
        s = chainer.Variable(self.sign, volatile=x.volatile)
        return functions.hadamard(x, t, s)
