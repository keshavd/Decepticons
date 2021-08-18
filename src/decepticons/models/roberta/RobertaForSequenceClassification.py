from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from decepticons.mixins.classification.SequenceClassificationMixin import (
    SequenceClassificationMixin,
)
from decepticons.interfaces.huggingface import HFClassificationInterface


class RobertaForSequenceClassification(
    SequenceClassificationMixin, RobertaPreTrainedModel, HFClassificationInterface
):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config=config)
        self.model = RobertaModel(config=config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config=config)

        self.init_weights()

    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs.last_hidden_state
        outputs.pooled_output = outputs.pooler_output
        return outputs
