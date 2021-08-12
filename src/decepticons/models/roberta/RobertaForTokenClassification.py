from transformers import RobertaPreTrainedModel, RobertaModel
from decepticons.mixins.classification.TokenClassificationMixin import (
    TokenClassificationMixin,
)
from decepticons.heads.TokenClassificationHead import (
    TokenClassificationHead,
)
from decepticons.interfaces.huggingface import HFClassificationInterface


class RobertaForTokenClassification(
    TokenClassificationMixin, RobertaPreTrainedModel, HFClassificationInterface
):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config=config)
        self.model = RobertaModel(config=config, add_pooling_layer=False)
        self.classifier = TokenClassificationHead(config=config)

        self.init_weights()

    def get_model_outputs(self, **kwargs):
        outputs = self.model(**kwargs)
        outputs.sequence_output = outputs[0]
        outputs.pooled_output = outputs[1]
        return outputs
