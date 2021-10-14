from dataclasses import dataclass
from typing import Union, Optional
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from datasets import Dataset


@dataclass
class DataCollatorForPairs(DataCollatorMixin):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        lookup_dataset: Dataset,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.lookup_dataset = lookup_dataset

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        a = [self.lookup_dataset[feature["a"]] for feature in features]
        b = [self.lookup_dataset[feature["b"]] for feature in features]
        batch_a = self.tokenizer.pad(
            a,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
        batch_b = self.tokenizer.pad(
            b,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
        full_batch = {
            "%s_a" % k: torch.tensor(v, dtype=torch.int64)
            for k, v in batch_a.items()
            if k in self.tokenizer.model_input_names
        }
        full_batch.update(
            {
                "%s_b" % k: torch.tensor(v, dtype=torch.int64)
                for k, v in batch_b.items()
                if k in self.tokenizer.model_input_names
            }
        )
        full_batch["labels"] = labels
        full_batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in full_batch.items()}
        return full_batch
