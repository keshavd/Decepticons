from transformers import BertPreTrainedModel, BertModel
from decepticons.mixins.classification.SequenceClassificationMixin import (
    SequenceClassificationMixin,
)
from decepticons.heads.BertSequenceClassificationHead import (
    BertSequenceClassificationHead,
)


class BertForSequenceClassification(BertPreTrainedModel, SequenceClassificationMixin):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = BertModel(config=config)
        self.classifier = BertSequenceClassificationHead(config=config)

    def get_model_outputs(self, **kwargs):
        outputs = self.model(**kwargs)
        outputs.sequence_output = outputs[0]
        outputs.pooled_output = outputs[1]
        return outputs
