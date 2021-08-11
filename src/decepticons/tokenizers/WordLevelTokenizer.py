from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.normalizers import Sequence, Lowercase, NFD
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from typing import Iterable


class WordLevelTokenizer(PreTrainedTokenizerFast):
    """
    Basic WordLevel Tokenizer

    This Tokenizer behaves like the original used in the first `SESAME` repository.
    """

    def __init__(self, training_iterable: Iterable = (), **kwargs):
        sesame_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        sesame_tokenizer.normalizer = Sequence([Lowercase(), NFD()])
        sesame_tokenizer.pre_tokenizer = WhitespaceSplit()
        sesame_tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        sesame_tokenizer.train_from_iterator(training_iterable, trainer)
        super().__init__(tokenizer_object=sesame_tokenizer, **kwargs)
        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.cls_token = "[CLS]"
