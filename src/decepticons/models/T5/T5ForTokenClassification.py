from decepticons.mixins.classification.TokenClassificationCrfMixin import (
    TokenClassificationCrfMixin,
)
from decepticons.heads.TokenClassificationCrfHead import TokenClassificationCrfHead
from transformers import T5EncoderModel
from decepticons.interfaces.huggingface import HFClassificationInterface


class T5ForTokenClassification(
    TokenClassificationCrfMixin, T5EncoderModel, HFClassificationInterface
):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **kwargs):
        super().__init__(config=config)
        self.model = T5EncoderModel(config=config)
        self.classifier = TokenClassificationCrfHead(config=config)
        self.num_labels = config.num_labels
        self.init_weights()

    def get_model_outputs(self, *args, **kwargs):
        """Get model output

        Note:
            T5 doesn't output a pooled output. Similar to implementation in
            Roberta, if using Pooled Output, just include pooling in the head.

        """
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs.last_hidden_state
        outputs.pooled_output = None
        return outputs
