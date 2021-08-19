from transformers import BertPreTrainedModel, BertModel
from decepticons.interfaces.huggingface import HFClassificationInterface
from decepticons.mixins.classification.SequenceClassificationMixin import (
    SequenceClassificationMixin,
)
from decepticons.heads.BertSequenceClassificationHead import (
    BertSequenceClassificationHead,
)


class BertForSequenceClassification(
    SequenceClassificationMixin, BertPreTrainedModel, HFClassificationInterface
):
    def __init__(self, config, **kwargs):
        super().__init__(config=config)
        self.model = BertModel(config=config)
        self.classifier = BertSequenceClassificationHead(config=config)
        self.num_labels = config.num_labels
        self.problem_type = None

    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs.last_hidden_state
        outputs.pooled_output = outputs.pooler_output
        return outputs
