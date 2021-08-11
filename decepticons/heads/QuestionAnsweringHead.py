from torch import nn


class QuestionAnsweringHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, sequence_output):
        return self.qa_outputs(sequence_output)
