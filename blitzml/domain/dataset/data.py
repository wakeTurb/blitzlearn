from abc import ABC
from dataclasses import dataclass

import numpy as np
import torch


class Abstract2DData(ABC):
    pass


@dataclass
class Numpy2DArray(Abstract2DData):
    values: np.ndarray

    def shape(self):
        return tuple(self.values.shape)


@dataclass
class Torch2DArray(Abstract2DData):
    values: torch.Tensor

    def shape(self):
        return tuple(self.values.shape)
