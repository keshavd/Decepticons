import torch
from typing import Union
from transformers import PreTrainedModel


def export_head_to_onnx(
    model: torch.nn.Module,
    input_example: Union[tuple, torch.Tensor],
    export_model_name: str,
    input_names=("input"),
    output_names=("output",),
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"},
    },
):
    """
    Uses Typical PyTorch export to export a model to ONNX.
    :param model:
    :param input_example:
    :param export_model_name:
    :return:
    """
    torch.onnx.export(
        model,
        input_example,
        export_model_name,
        export_params=True,
        opset_version=10,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("ONNX export complete at {}".format(export_model_name))
    return 0


def export_base_model_to_onnx(
    model: PreTrainedModel,
    input_example: Union[tuple, torch.Tensor],
    export_model_name: str,
    input_names=("input_ids", "token_type_ids", "attention_mask"),
    output_names=("embedding_input",),
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "embedding_input": {0: "batch_size", 1: "seq_len"},
    },
):
    """

    :param model:
    :param input_example:
    :param output_names:
    :param export_model_name:
    :param input_names:
    :return:
    """
    torch.onnx.export(
        model,
        input_example,
        export_model_name,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("ONNX export complete at {}".format(export_model_name))
    return 0
