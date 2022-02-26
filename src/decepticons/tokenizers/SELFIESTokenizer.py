from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.decoders import Decoder
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from typing import Iterable
from decepticons.tokenizers.decoders.SELFIESDecoder import SELFIESDecoder
from decepticons.tokenizers.pre_tokenizers.SELFIESPreTokenizer import (
    SELFIESPreTokenizer,
)
from tokenizers.normalizers import Normalizer
from decepticons.tokenizers.normalizers.RemoveChiralityNormalizer import (
    RemoveChiralityNormalizer,
)


class SELFIESTokenizer(PreTrainedTokenizerFast):
    """
    Converts a SMILES string into a SELFIES string.
    """

    def __init__(self, training_iterable: Iterable = (), **kwargs):
        selfies_tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        selfies_tokenizer.normalizer = Normalizer.custom(RemoveChiralityNormalizer())
        selfies_tokenizer.pre_tokenizer = PreTokenizer.custom(SELFIESPreTokenizer())
        selfies_tokenizer.decoder = Decoder.custom(SELFIESDecoder())
        selfies_tokenizer.post_processor = TemplateProcessing(
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
        selfies_tokenizer.train_from_iterator(training_iterable, trainer)
        super().__init__(tokenizer_object=selfies_tokenizer, **kwargs)
        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.cls_token = "[CLS]"
