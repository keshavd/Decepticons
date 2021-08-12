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

    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs[0]
        outputs.pooled_output = outputs[1]
        return outputs
