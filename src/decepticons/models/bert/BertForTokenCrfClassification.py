from transformers import BertPreTrainedModel, BertModel
from decepticons.interfaces.huggingface import HFClassificationInterface
from decepticons.mixins.classification.TokenClassificationCrfMixin import (
    TokenClassificationCrfMixin,
)
from decepticons.heads.TokenClassificationCrfHead import (
    TokenClassificationCrfHead,
)


class BertForTokenCrfClassification(
    TokenClassificationCrfMixin, BertPreTrainedModel, HFClassificationInterface
):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    def __init__(self, config, ignore_index=None, **kwargs):
        super().__init__(config=config)
        self.model = BertModel(config=config, add_pooling_layer=False)
        self.classifier = TokenClassificationCrfHead(config=config)
        self.num_labels = config.num_labels
        self.ignore_index = ignore_index
        self.init_weights()
    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs.last_hidden_state
        outputs.pooled_output = outputs.pooler_output
        return outputs
