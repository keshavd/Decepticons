from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class MultipleClassificationHeadModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[dict] = None
