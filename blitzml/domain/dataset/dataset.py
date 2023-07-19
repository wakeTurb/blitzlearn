from dataclasses import dataclass
from typing import List

from .data import Abstract2DData


@dataclass
class Dataset:
    value: Abstract2DData
    feats: List[str]
