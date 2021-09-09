import selfies as sf
from typing import List


class SELFIESDecoder:
    def decode(self, tokens: List[str]) -> str:
        return sf.decoder("".join(tokens))
