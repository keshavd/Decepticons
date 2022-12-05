from dataclasses import dataclass
from typing import Optional, Union
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)
import torch
from torch.nn import functional as F
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorForT5MLM(DataCollatorMixin):
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def torch_call(self, features):
        # convert list to dict and tensorize input
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        input_ids = batch["input_ids"]
        batch_size, expanded_input_length = input_ids.shape

        mask_indices = torch.stack(
            [
                self.random_spans_noise_mask(expanded_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices)
        labels_sentinel = self.create_sentinel_ids(labels_mask)

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = self.shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - torch.roll(mask_indices, 1, -1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = torch.where(
            start_indices != 0, torch.cumsum(start_indices, -1), start_indices
        )
        sentinel_ids = torch.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = torch.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = torch.concat(
            [
                input_ids,
                torch.full((batch_size, 1), self.tokenizer.eos_token_id),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length
        num_noise_tokens = torch.round(torch.tensor(orig_length * 0.2))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = torch.round(
            torch.tensor(num_noise_tokens / self.mean_noise_span_length)
        )
        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = torch.arange(num_items - 1) < (num_segments - 1)
            idx = torch.randperm(mask_indices.nelement())
            mask_indices = mask_indices.view(-1)[idx].view(mask_indices.size())
            first_in_segment = F.pad(mask_indices, ((1, 0)))
            segment_id = torch.cumsum(first_in_segment, 0)
            # count length of sub-segments assuming that list is sorted
            _, segment_length = torch.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )
        interleaved_span_lengths = torch.reshape(
            torch.stack([nonnoise_span_lengths, noise_span_lengths], 1),
            (int(num_noise_spans * 2),),
        )
        span_starts = torch.cumsum(interleaved_span_lengths[:-1], 0)
        span_start_indicator = torch.zeros((length,))
        span_start_indicator[span_starts] = True
        span_num = torch.cumsum(span_start_indicator, 0)
        is_noise = torch.eq(span_num % 2, torch.tensor(1)).long()
        return is_noise[:orig_length]

    # Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
    def shift_tokens_right(
        self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
    ) -> torch.Tensor:
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = torch.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id
        shifted_input_ids = torch.where(
            shifted_input_ids == -100, pad_token_id, shifted_input_ids
        )
        return shifted_input_ids
