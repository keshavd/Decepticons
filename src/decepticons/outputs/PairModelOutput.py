from dataclasses import dataclass
import torch
from typing import Optional, Tuple


@dataclass
class PairModelOutput:
    hidden_states_a: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_b: Optional[Tuple[torch.FloatTensor]] = None
