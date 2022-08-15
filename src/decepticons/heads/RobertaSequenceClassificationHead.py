import torch
from torch import nn


class RobertaNoClassificationHead(nn.Module):
    """Head for a split classification layer. This is the hidden state."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, s_token, **kwargs):
        x = self.dropout(s_token)
        x = self.dense(x)
        x = torch.tanh(x)
        return x


class RobertaOnlyClassificationHead(nn.Module):
    """Head for a split classification Layer. This is the classifier."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, projection, **kwargs):
        x = self.dropout(projection)
        x = self.out_proj(x)
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # Hidden Projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output Projection
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # Get Hidden Projection
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # Get Output Projection
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaFlexibleClassificationHead(nn.Module):
    """Head for sentence-level classification tasks with custom index."""

    def __init__(self, config, cls_token_idx=0):
        super().__init__()
        # Hidden Projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output Projection
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_token_idx = cls_token_idx

    def forward(self, features, **kwargs):
        # Get Hidden Projection
        x = features[:, self.cls_token_idx, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # Get Output Projection
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
