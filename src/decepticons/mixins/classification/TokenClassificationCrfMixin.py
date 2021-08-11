from abc import ABC
import torch
from transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput


class TokenClassificationCrfMixin(PreTrainedModel, ABC):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_labels = config.num_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.get_model_outputs(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        emissions = outputs.hidden_states[0]
        log_likelihood = self.classifier(
            emissions=emissions, tags=labels, mask=attention_mask
        )
