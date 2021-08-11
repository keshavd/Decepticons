from transformers import BertPreTrainedModel, BertModel
from src.decepticons import (
    TokenClassificationMixin,
)
from src.decepticons import (
    TokenClassificationHead,
)


class BertForTokenClassification(BertPreTrainedModel, TokenClassificationMixin):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BertModel(config=config, add_pooling_layer=False)
        self.classifier = TokenClassificationHead(config=config)

        self.init_weights()

    def get_model_outputs(self, **kwargs):
        outputs = self.model(**kwargs)
        outputs.sequence_output = outputs[0]
        outputs.pooled_output = outputs[1]
        return outputs
