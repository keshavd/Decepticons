from transformers import PreTrainedTokenizerFast
from tokenizers.models import WordLevel
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString
from tokenizers.normalizers import Sequence, Lowercase, NFD
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.decoders import Decoder
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
import selfies as sf
from typing import List, Iterable


class SELFIESPreTokenizer:
    def selfies_split(
        self, i: int, normalized_str: NormalizedString
    ) -> List[NormalizedString]:
        splits = list(sf.split_selfies(sf.encoder(str(normalized_str))))
        return [NormalizedString(x) for x in splits]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.selfies_split)


class SELFIESDecoder:
    def decode(self, tokens: List[str]) -> str:
        return sf.decoder("".join(tokens))


class SELFIESTokenizer(PreTrainedTokenizerFast):
    def __init__(self, training_iterable: Iterable = (), **kwargs):
        selfies_tokenzier = Tokenizer(WordLevel(unk_token="[UNK]"))
        selfies_tokenzier.normalizer = Sequence([Lowercase(), NFD()])
        selfies_tokenzier.pre_tokenizer = PreTokenizer.custom(SELFIESPreTokenizer())
        selfies_tokenzier.decoder = Decoder.custom(SELFIESDecoder())
        selfies_tokenzier.post_processor = TemplateProcessing(
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
        selfies_tokenzier.train_from_iterator(training_iterable, trainer)
        super().__init__(tokenizer_object=selfies_tokenzier, **kwargs)
        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.cls_token = "[CLS]"
