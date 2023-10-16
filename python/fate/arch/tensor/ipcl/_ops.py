import torch
from fate.arch.tensor import _custom_ops

from ._tensor import IpclTensor, implements


@implements(_custom_ops.decrypt_f)
def decrypt_f(input, decryptor):
    return decryptor.decrypt(input)


@implements(torch.add)
def add(input: IpclTensor, other):
    if not isinstance(input, IpclTensor) and isinstance(other, IpclTensor):
        return add(other, input)

    if isinstance(other, IpclTensor) and input.shape == other.shape:
        return IpclTensor(input._data + other._data, torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, torch.Tensor) and input.shape == other.shape:
        return IpclTensor(input._data + other.detach().numpy().flatten(), torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, (float, int)):
        return IpclTensor(input._data + other, torch.promote_types(input.dtype, type(other)), input._shape)
    return NotImplemented


@implements(torch.rsub)
def rsub(input, other):
    if not isinstance(input, IpclTensor) and isinstance(other, IpclTensor):
        return sub(other, input)

    if isinstance(other, IpclTensor) and input.shape == other.shape:
        return IpclTensor(other._data - input._data, torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, torch.Tensor) and input.shape == other.shape:
        return IpclTensor(input._data.__rsub__(other.detach().numpy().flatten()), torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, (float, int)):
        return IpclTensor(input._data.__rsub__(other), torch.promote_types(input.dtype, type(other)), input._shape)
    return NotImplemented


@implements(torch.sub)
def sub(input, other):
    if not isinstance(input, IpclTensor) and isinstance(other, IpclTensor):
        return rsub(other, input)

    if isinstance(other, IpclTensor) and input.shape == other.shape:
        return IpclTensor(input._data - other._data, torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, torch.Tensor) and input.shape == other.shape:
        return IpclTensor(input._data - other.detach().numpy().flatten(), torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, (float, int)):
        return IpclTensor(input._data - other, torch.promote_types(input.dtype, type(other)), input._shape)
    return NotImplemented


@implements(torch.mul)
def mul(input, other):
    # assert input is IpclTensor
    if not isinstance(input, IpclTensor) and isinstance(other, IpclTensor):
        return mul(other, input)

    if isinstance(other, IpclTensor):
        raise ValueError("can't mul `IpclTensor` with `IpclTensor`")
    if isinstance(other, torch.Tensor) and input.shape == other.shape:
        return IpclTensor(input._data * other.detach().numpy().flatten(), torch.promote_types(input.dtype, other.dtype), input._shape)
    if isinstance(other, (float, int)):
        return IpclTensor(input._data * other, torch.promote_types(input.dtype, type(other)), input._shape)
    return NotImplemented


@implements(_custom_ops.rmatmul_f)
def rmatmul_f(input, other):
    if not isinstance(input, IpclTensor) and isinstance(other, IpclTensor):
        return matmul(other, input)

    if isinstance(other, torch.Tensor):
        other_size = other.shape[0] if len(
            other.shape) == 1 else other.shape[0] * other.shape[1]
        output_shape = 1 if input.shape[0] == other_size else other.shape[0]
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if len(input.shape) == 1:
            return IpclTensor(input._data.rmatmul_f(other.detach().numpy()), output_dtype, (output_shape,))
        elif len(input.shape) == 2:
            return IpclTensor(input._data.rmatmul_f(other.detach().numpy()), output_dtype, (output_shape, input.shape[1]))
        else:
            raise ValueError(
                f"can't matmul `IpclTensor` with `torch.Tensor` with dim `{len(other.shape)}`")
    return NotImplemented


@implements(torch.matmul)
def matmul(input, other):
    if not isinstance(input, IpclTensor) and isinstance(other, IpclTensor):
        return rmatmul_f(other, input)

    if isinstance(other, IpclTensor):
        raise ValueError("can't matmul `IpclTensor` with `IpclTensor`")

    if isinstance(other, torch.Tensor):
        input_size = input.shape[0] if len(
            input.shape) == 1 else input.shape[0] * input.shape[1]
        output_shape = 1 if other.shape[0] == input_size else input.shape[0]
        output_dtype = torch.promote_types(input.dtype, other.dtype)
        if len(other.shape) == 1:
            return IpclTensor(input._data.matmul(other.detach().numpy()), output_dtype, (output_shape,))
        elif len(other.shape) == 2:
            return IpclTensor(input._data.matmul(other.detach().numpy()), output_dtype, (output_shape, other.shape[1]))
        else:
            raise ValueError(
                f"can't matmul `IpclTensor` with `torch.Tensor` with dim `{len(other.shape)}`")
    return NotImplemented


@implements(_custom_ops.to_local_f)
def to_local_f(input):
    return input
