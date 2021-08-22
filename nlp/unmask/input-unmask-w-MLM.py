# from https://huggingface.co/transformers/v2.8.0/usage.html#masked-language-modeling
import os
import numpy as np
from transformers import BertForMaskedLM, AutoTokenizer
import torch

dataset_iter = None
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
sequence = f"Distilled models are much smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint in a very big way."


def cleanup():
    # Release the memory
    global dataset_iter
    dataset_iter = None


def get_input_activations(input_shapes=[[1, 128]], ):
    global dataset_iter
    batch_size = input_shapes[0][0]

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding='max_length', model_max_length=128)
    
    print('SEQUENCE:\n', sequence)
    input = tokenizer.encode(sequence, return_tensors="pt")
    pad = torch.zeros(128 - input.shape[1], dtype=torch.long).unsqueeze(0) # assuning pad token is 0
    input = torch.cat((input, pad), 1)

    print('FINAL INPUT\n', input)
    print('FINAL INPUT SHAPE\n', input.shape)

    ret = [input.detach().numpy().astype(np.int64)]

    return ret
