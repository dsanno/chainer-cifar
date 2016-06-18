import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

class Arrange(function.Function):

    """Arrange vector."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[1] == t_type.shape[0],
        )

    def forward(self, inputs):
        x, t = inputs
        return x.take(t, axis=1),

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gy = grad_outputs[0]
        gx = numpy.zeros_like(x)
        gx[:, t] = gy
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        x, t = inputs
        gy = grad_outputs[0]
        gx = cuda.cupy.zeros_like(x)
        gx = cuda.elementwise(
            'S t, raw T gy, int32 c',
            'raw T gx',
            '''
            for (int j = 0; j < c; j++) {
              int ind_gx[] = {j, t};
              int ind_gy[] = {j, i};
              gx[ind_gx] = gy[ind_gy];
            }
            ''',
            'arrange_bwd'
        )(t, gy, gy.shape[0], gx)
        return gx, None

def arrange(x, t):
    """arrange vector.

    This function returns ``x.take(t)``

    Args:
        x (Variable): Variable storing arrays.
        t (Variable): Variable storing index numbers.

    Returns:
        ~chainer.Variable: Variable arranged ``x`` indexed by `t`

    """
    return Arrange()(x, t)

class Hadamard(function.Function):

    """Hadamard transform."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, t_type, s_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 2,
            x_type.shape[1] & (x_type.shape[1] - 1) == 0,
            t_type.dtype.kind == 'i',
            t_type.ndim == 2,
            t_type.shape[1] == x_type.shape[1],
            s_type.dtype.kind == 'f',
            s_type.ndim == 2,
            s_type.shape[1] == x_type.shape[1],
        )

    def forward(self, inputs):
        x, t, s = inputs
        size = x.shape[1]
        y = x
        i = 0
        while size > 1:
            next_size = size // 2
            y = y * s[i] + y.take(t[i], axis=1)
            size = next_size
            i += 1
        return y,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        x, t, s = inputs
        size = x.shape[1]
        gx = gy
        i = 0
        while size > 1:
            next_size = size // 2
            gx = gx * s[i] + gx.take(t[i], axis=1)
            size = next_size
            i += 1
        return gx, None, None

def hadamard(x, t, s):
    """Hadamard transform.

    Args:
        x (Variable): Variable storing arrays.
        t (Variable): Variable transpose for Hadamard transform.
        s (Variable): Variable sign for Hadamard transform.

    Returns:
        ~chainer.Variable: Variable transformed value.

    """
    return Hadamard()(x, t, s)
