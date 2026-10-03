from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd


@dataclass
class NodeSamples:
    indices: pd.DataFrame
    values: dict[str, np.ndarray]


Simulation: TypeAlias = dict[str, NodeSamples]
