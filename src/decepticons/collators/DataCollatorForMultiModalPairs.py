from dataclasses import dataclass
from typing import Union, Optional
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from datasets import Dataset


@dataclass
class DataCollatorForMultiModalPairs(DataCollatorMixin):
    def __init__(
        self,
        tokenizer_0: PreTrainedTokenizerBase,
        lookup_dataset_0: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        lookup_dataset: Dataset,
        max_length_0: Optional[int] = None,
        label_pad_token_id_0: int = -100,
        max_length: Optional[int] = None,
        label_pad_token_id: int = -100,
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        symmetric_relationship: bool = True,
    ):
        self.lookup_dataset = lookup_dataset
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.symmetric_relationship = symmetric_relationship
        # Second Data Type
        self.lookup_dataset_0 = lookup_dataset_0
        self.tokenizer_0 = tokenizer_0
        self.max_length_0 = max_length_0
        self.label_pad_token_id_0 = label_pad_token_id_0

    def torch_call(self, features):
        import torch
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        a = [self.lookup_dataset[feature["a"]] for feature in features]
        b = [self.lookup_dataset_0[feature["b"]] for feature in features]

        if self.symmetric_relationship:
            a += [self.lookup_dataset[feature["b"]] for feature in features]
            b += [self.lookup_dataset_0[feature["a"]] for feature in features]
            labels += labels

        batch_a = self.tokenizer.pad(
            a,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
        batch_b = self.tokenizer_0.pad(
            b,
            padding=self.padding,
            max_length=self.max_length_0,
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
                if k in self.tokenizer_0.model_input_names
            }
        )
        full_batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return full_batch
