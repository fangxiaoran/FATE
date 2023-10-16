import functools
import math

import torch

_HANDLED_FUNCTIONS = {}


class IpclTensor:
    def __init__(self, data, dtype, shape) -> None:
        self._data = data
        self._dtype = dtype
        self._shape = shape

    def __repr__(self) -> str:
        return f"<IpclTensor shape={self.shape}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, item):
        if isinstance(item, int):
            return IpclTensor(self._data.__getitem__(item), self._dtype, (1,))
        elif isinstance(item, slice):
            step = 1 if not item.step else item.step
            n = math.ceil((item.stop - item.start) / step)
            return IpclTensor(self._data.__getitem__(item), self._dtype, (n,))
        else:
            raise NotImplementedError(f"item {item} not supported")

    @property
    def shape(self):
        return torch.Size(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(issubclass(t, (torch.Tensor, IpclTensor)) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    """implement arth magic"""

    def __add__(self, other):
        from ._ops import add

        return add(self, other)

    def __radd__(self, other):
        from ._ops import add

        return add(other, self)

    def __sub__(self, other):
        from ._ops import sub

        return sub(self, other)

    def __rsub__(self, other):
        from ._ops import rsub

        return rsub(self, other)

    def __mul__(self, other):
        from ._ops import mul

        return mul(self, other)

    def __rmul__(self, other):
        from ._ops import mul

        return mul(other, self)

    def __matmul__(self, other):
        from ._ops import matmul

        return matmul(self, other)

    def __rmatmul__(self, other):
        from ._ops import rmatmul_f

        return rmatmul_f(self, other)


def implements(torch_function):
    """Register a torch function override for MockPaillierTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator
