from abc import ABC
import torch
from transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput


class TokenClassificationCrfMixin(PreTrainedModel, ABC):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_labels = config.num_labels
