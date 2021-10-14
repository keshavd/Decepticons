import torch
from torch import nn
from transformers.models.longformer.modeling_longformer import (
    LongformerClassificationHead,
)


class LongformerPairedClassificationHead(LongformerClassificationHead):
    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, hidden_states_a, hidden_states_b, **kwargs):
        hidden_states = torch.cat([hidden_states_a[:, 0, :], hidden_states_b[:, 0, :]])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output
