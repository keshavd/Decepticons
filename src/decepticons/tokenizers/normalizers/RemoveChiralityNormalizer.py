from tokenizers import Tokenizer, Regex, NormalizedString, PreTokenizedString
from rdkit import Chem


class RemoveChiralityNormalizer:
    """
    Normalizer for removing chirality from smiles strings.
    """

    def normalize(self, normalized: NormalizedString):
        mol = Chem.MolFromSmiles(normalized.original)
        non_chiral_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        return normalized.replace(normalized.original, non_chiral_smiles)
