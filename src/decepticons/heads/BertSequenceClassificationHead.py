from torch import nn, onnx, randn


class BertSequenceClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, "classifier_dropout")
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output, **kwargs):
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return x
