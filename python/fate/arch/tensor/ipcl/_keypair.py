from ipcl_python import PaillierKeypair
import torch

from ._tensor import IpclTensor


def keygen(key_length):
    pk, sk = PaillierKeypair.generate_keypair(key_length)
    return IpclTensorEncryptor(pk), IpclTensorDecryptor(sk)


class IpclTensorEncryptor:
    def __init__(self, key) -> None:
        self._key = key

    def encrypt(self, tensor: torch.Tensor):
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().numpy()
            return IpclTensor(self._key.encrypt(array.flatten()), tensor.dtype, array.shape)
        elif hasattr(tensor, "encrypt"):
            return tensor.encrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")


class IpclTensorDecryptor:
    def __init__(self, key) -> None:
        self._key = key

    def decrypt(self, tensor: IpclTensor):
        if isinstance(tensor, IpclTensor):
            return torch.tensor(self._key.decrypt(tensor._data)).reshape(tensor.shape)
        elif hasattr(tensor, "decrypt"):
            return tensor.decrypt(self)
        raise NotImplementedError(f"`{tensor}` not supported")
