from transformers import PreTrainedModel
from better_abc import ABCMeta, abstract_attribute, abstractmethod


class HFModelInterface(PreTrainedModel, metaclass=ABCMeta):
    @abstract_attribute
    def model(self):
        pass

    @abstractmethod
    def get_model_outputs(self, **kwargs):
        pass


class HFClassificationInterface(HFModelInterface, metaclass=ABCMeta):
    def classifier(self):
        pass
