import torch
from torch import nn
from torchcrf import CRF
import torch.nn.functional as F


class TokenClassificationCrfHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels)

    def forward(self, sequence_output, tags, mask=None, reduction="sum"):
        """ Returns negative log-likelihood"""
        x = self.dropout(sequence_output)
        x = self.classifier(x)
        x = self.crf(emissions=x, mask=mask, tags=tags, reduction=reduction)
        return torch.neg(x)

    def predict(self, sequence_output, mask=None):
        """ Returns the calculated labels via viterbi algo"""
        x = self.classifier.eval()(sequence_output)
        x = self.crf.decode(emissions=x, mask=mask)
        return F.one_hot(x, num_classes=self.crf.num_tags)
