from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from src.decepticons.mixins.classification.SequenceClassificationMixin import (
    SequenceClassificationMixin,
)


class RobertaForSequenceClassification(
    RobertaPreTrainedModel, SequenceClassificationMixin
):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self._model = RobertaModel(config=config, add_pooling_layer=False)
        self._classifier = RobertaClassificationHead(config=config)

        self.init_weights()

    def get_model_outputs(self, **kwargs):
        outputs = self.model(**kwargs)
        outputs.sequence_output = outputs[0]
        outputs.pooled_output = outputs[1]
        return outputs

    def model(self):
        return self._model
