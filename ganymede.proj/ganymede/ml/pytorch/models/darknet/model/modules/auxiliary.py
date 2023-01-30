# python
from typing import Tuple
# 3rd party
import torch
import numpy as np
from ganymede.rw import BinaryPackageReader


def read_tensor_from_binary(
    reader : BinaryPackageReader,
    shape  : Tuple[int, ...],
    device : torch.device
) -> torch.Tensor:
    if len(shape) < 1:
        raise ValueError(f'Invalid shape for tensor reading from binary:{len(shape)}')

    float_size = 4
    total_size = int(np.prod(shape))

    bin_array = reader.read_bytes(total_size * float_size)
    np_array  = np.frombuffer(bin_array, np.float32)

    np_array.shape = shape

    data_t = torch.from_numpy(np_array).to(device)

    return data_t