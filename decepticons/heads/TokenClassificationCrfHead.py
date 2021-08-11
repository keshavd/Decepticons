from torch import nn
from torchcrf import CRF


class TokenClassificationCrfHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier = CRF(config.vocab_size)

    def forward(self, hidden_states):
        x = self.classifier(hidden_states)
        return x
