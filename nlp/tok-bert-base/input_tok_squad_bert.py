# from https://huggingface.co/transformers/v2.8.0/usage.html#extractive-question-answering
# https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/huggingface_pytorch-transformers.ipynb
import os

import torch
import numpy as np
from transformers import AutoTokenizer
from testify.tests import get_full_path_to_test_file

dataset_iter = None


text = "Jim Henson was a puppeteer"
question = "Who was Jim Henson ?"


def cleanup():
    # Release the memory
    global dataset_iter
    dataset_iter = None


def get_input_activations(
    input_shapes=[[1, 128]],
    filename="/home/software/mount/data/software/graph_compiler/activation_data/squad/squad_bert.pt",
    return_token_type_ids=True,
    return_attention_mask=False,
):
    global dataset_iter
    assert (
        len(input_shapes) == 1
    ), f"Expecting only one input shape request for bert squad, got: {input_shapes}"
    assert (
        len(input_shapes[0]) == 2
    ), "Expecting 2D shape request for bert squad, of shape (batch_size, sequence_len)"
    batch_size = input_shapes[0][0]
    model_name = "bert-large-cased-whole-word-masking-finetuned-squad"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('TEXT:\n', text)
    print('QUESTION:\n', question)
    inputs = tokenizer.encode_plus(question, text,  
                                   add_special_tokens=True,
                                   return_tensors="pt")
    input_data = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    print('input data:\n', input_data)
    print('token_type_ids:\n', token_type_ids)

    ret = [input_data.detach().numpy().astype(np.int64)]
    if return_token_type_ids:
        ret.append(token_type_ids.detach().numpy().astype(np.int64))
    if return_attention_mask:
        ret.append(attention_mask.detach().numpy().astype(np.int64))
    return ret
