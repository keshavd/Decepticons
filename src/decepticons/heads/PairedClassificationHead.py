from torch import nn, onnx, randn
import torch


class PairedClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout")
            else config.hidden_dropout_prob
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states_a, hidden_states_b, **kwargs):
        hidden_states = torch.cat(
            [hidden_states_a[:, 0, :], hidden_states_b[:, 0, :]], 1
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.classifier(hidden_states)
        return output
