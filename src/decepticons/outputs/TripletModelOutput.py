from dataclasses import dataclass
import torch
from typing import Optional,Tuple


@dataclass
class TripletModelOutput:
    hidden_states_anchor: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_positive: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_negative: Optional[Tuple[torch.FloatTensor]] = None
