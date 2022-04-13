import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import PreTrainedModel


class TokenClassificationCrfMixin(PreTrainedModel):
    """Performs Token-level CRF Classification with model's `sequence_output`"""

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
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
        sequence_output = outputs.sequence_output
        loss = self.classifier.get_loss(
            sequence_output=sequence_output, tags=labels, mask=attention_mask > 0
        )
        # Made up the logits (its just one-hot encoded labels)
        logits = self.classifier.predict(sequence_output, mask=attention_mask > 0)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
