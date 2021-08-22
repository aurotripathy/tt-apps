import os

import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
from testify.tests import get_full_path_to_test_file

dataset_iter = None


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
    input_data = torch.tensor([[  101, 19081,  3640,  6970, 25918,  8010,  2090,  2029,  7705,  2015,
                                  1029,   102,   100, 19081,  1006,  3839,  2124,  2004,  1052, 22123,
                                  2953,  2818,  1011, 19081,  1998,  1052, 22123,  2953,  2818,  1011,
                                  3653, 23654,  2098,  1011, 14324,  1007,  3640,  2236,  1011,  3800,
                                  4294,  2015,  1006, 14324,  1010, 14246,  2102,  1011,  1016,  1010,
                                  23455,  1010, 28712,  2213,  1010,  4487, 16643, 23373,  1010, 28712,
                                  7159,  1529,  1007,  2005,  3019,  2653,  4824,  1006, 17953,  2226,
                                  1007,  1998,  3019,  2653,  4245,  1006, 17953,  2290,  1007,  2007,
                                  2058,  3590,  1009,  3653, 23654,  2098,  4275,  1999,  2531,  1009,
                                  4155,  1998,  2784,  6970, 25918,  8010,  2090, 23435, 12314,  1016,
                                  1012,  1014,  1998,  1052, 22123,  2953,  2818,  1012,   102]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    print('input data\n', input_data)
    print('token_type_ids\n', token_type_ids)

    ret = [input_data.detach().numpy().astype(np.int64)]
    if return_token_type_ids:
        ret.append(token_type_ids.detach().numpy().astype(np.int64))
    if return_attention_mask:
        ret.append(attention_mask.detach().numpy().astype(np.int64))
    return ret
