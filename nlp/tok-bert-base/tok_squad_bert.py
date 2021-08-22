import os

import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler

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

    if dataset_iter is None:
        # Load the full data set, and randomly sample something
        try:
            if os.path.exists("/tenstorrent/releases"):
                # Release mode paths are different
                filename = get_full_path_to_test_file(
                    "releases/tests/activations/data/squad_bert.pt"
                )
            else:
                filename = get_full_path_to_test_file(filename)
        except:
            filename = "/home/software/mount/data/software/graph_compiler/activation_data/squad/squad_bert.pt"  # fall-back

        print(f"Loading tokenized SQuAD data from {filename}")
        data = torch.load(filename)
        _, dataset, _ = (
            data["features"],
            data["dataset"],
            data["examples"],
        )
        eval_sampler = RandomSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=batch_size
        )
        dataset_iter = eval_dataloader.__iter__()
        print("Done.")

    full_data = dataset_iter.__next__()
    input_data = full_data[0]
    attention_mask = full_data[1]
    token_type_ids = full_data[2]

    ret = [input_data.detach().numpy().astype(np.int64)]
    if return_token_type_ids:
        ret.append(token_type_ids.detach().numpy().astype(np.int64))
    if return_attention_mask:
        ret.append(attention_mask.detach().numpy().astype(np.int64))
    return ret
