from .mock import keygen as mock_keygen
from .paillier import keygen as paillier_keygen
from .ipcl import keygen as ipcl_keygen


def phe_keygen(kind, options):
    if kind == "paillier":
        return paillier_keygen(**options)
    elif kind == "ipcl":
        return ipcl_keygen(**options)
    elif kind == "mock":
        return mock_keygen(**options)
    else:
        raise ValueError(f"Unknown PHE keygen kind: {kind}")
