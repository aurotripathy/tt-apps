# https://huggingface.co/transformers/v2.8.0/usage.html#masked-language-modeling
import torch
from torch import nn
from transformers.modeling_bert import BertForMaskedLM
from transformers import AutoTokenizer

import sage.api as tt_api

model_name = 'bert-base-cased'
def get_model(
    par_strategy=None, core_count=12, num_layers=None,
        pretrained_name=model_name
):
    class UnmaskWithMLM(nn.Module):
        def __init__(self):
            super(UnmaskWithMLM, self).__init__()
            pretrained_name = model_name

            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)                
            self.model = BertForMaskedLM.from_pretrained(pretrained_name).eval()


        def forward(self, inputs):
            print('Inputs\n', inputs)
            token_logits = self.model(inputs)[0]
            print('Output shape:\n', token_logits.shape)
            
            mask_token_index = torch.where(inputs == self.tokenizer.mask_token_id)[1]
            mask_token_logits = token_logits[0, mask_token_index, :]

            top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            print(top_5_tokens)

            for token in top_5_tokens:
                print(self.tokenizer.mask_token, self.tokenizer.decode([token]))
            
            # return token_logits
            return token_logits

    model = UnmaskWithMLM()
    
    # Set parallelization strategy
    tt_api.set_parallelization_strat(
        model, cores=((0, 0), (core_count - 1, 0)), strategy=par_strategy
    )

    tt_api.set_parallelization_strat(
        model.model.bert.embeddings, strategy="RowParallel"
    )

    tt_api.set_parallelization_strat(
        model.model.bert.embeddings.LayerNorm, strategy=par_strategy
    )

    if par_strategy is None:
        pass

    return model
