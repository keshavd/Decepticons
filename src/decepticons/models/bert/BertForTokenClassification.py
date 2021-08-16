from transformers import BertPreTrainedModel, BertModel
from decepticons.interfaces.huggingface import HFClassificationInterface
from decepticons.mixins.classification.TokenClassificationMixin import (
    TokenClassificationMixin,
)
from decepticons.heads.TokenClassificationHead import (
    TokenClassificationHead,
)


class BertForTokenClassification(
    TokenClassificationMixin, BertPreTrainedModel, HFClassificationInterface
):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **kwargs):
        super().__init__(config=config)
        self.model = BertModel(config=config, add_pooling_layer=False)
        self.classifier = TokenClassificationHead(config=config)
        self.num_labels = config.num_labels

        self.init_weights()

    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs.last_hidden_state
        outputs.pooled_output = outputs.pooler_output
        return outputs
