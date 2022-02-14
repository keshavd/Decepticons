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
        self.num_labels = config.num_labels
        self.problem_type = None

        self.init_weights()

    def get_model_outputs(self, *args, **kwargs):
        """Modification of the BaseModelOutputWithPoolingAndCrossAttentions object

        We're going to pass in the hidden_state as the pooled output because
        `RobertaClassificationHead` will calculate a pooled-output representation
        of the [CLS] token

        see link below for details:
            https://github.com/huggingface/transformers/issues/8776#issuecomment-733557182
        """
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs.last_hidden_state
        outputs.pooled_output = outputs.last_hidden_state
        return outputs


class RobertaForSequenceClassificationWithActiveLearning(
    RobertaForSequenceClassification
):
    def __init__(self, config, **kwargs):
        from baal.bayesian.dropout import patch_module

        super().__init__(config=config, **kwargs)
        self.model = patch_module(self.model)
        self.classifier = patch_module(self.classifier)
