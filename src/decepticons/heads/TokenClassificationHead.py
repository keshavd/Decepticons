from torch import nn


class TokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, sequence_output):
        x = self.dropout(sequence_output)
        x = self.classifier(x)
        return x
