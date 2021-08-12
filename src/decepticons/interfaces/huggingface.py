from abc import ABC, abstractmethod


class HFModelInterface(ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.config = None

    @abstractmethod
    def get_model_outputs(self, *args, **kwargs):
        pass


class HFClassificationInterface(HFModelInterface, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None
        self.num_labels = None
        self.problem_type = None


class HFQuestionAnsweringInterface(HFModelInterface, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qa_outputs = None
