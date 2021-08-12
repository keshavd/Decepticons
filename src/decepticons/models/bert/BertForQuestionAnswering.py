from transformers import BertPreTrainedModel, BertModel
from decepticons.interfaces.huggingface import HFQuestionAnsweringInterface
from decepticons.mixins.answering.QuestionAnsweringMixin import (
    QuestionAnsweringMixin,
)
from decepticons.heads.QuestionAnsweringHead import (
    QuestionAnsweringHead,
)


class BertForQuestionAnswering(
    QuestionAnsweringMixin, BertPreTrainedModel, HFQuestionAnsweringInterface
):
    def __init__(self, config, **kwargs):
        super().__init__(config=config)
        self.model = BertModel(config=config)
        self.qa_outputs = QuestionAnsweringHead(config=config)

    def get_model_outputs(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        outputs.sequence_output = outputs[0]
        outputs.pooled_output = outputs[1]
        return outputs
