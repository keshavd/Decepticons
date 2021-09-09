from tokenizers import NormalizedString, PreTokenizedString
from typing import List
import selfies as sf


class SELFIESPreTokenizer:
    def selfies_split(
        self, i: int, normalized_str: NormalizedString
    ) -> List[NormalizedString]:

        splits = list(sf.split_selfies(sf.encoder(str(normalized_str))))
        return [NormalizedString(x) for x in splits]

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.selfies_split)
