from torch import nn
from transformers.modeling_bert import BertConfig, BertForQuestionAnswering

import sage.api as tt_api


def get_model(
    par_strategy=None, core_count=12, num_layers=None, pretrained_name="bert-base-cased"
):
    class BertQA(nn.Module):
        def __init__(self):
            super(BertQA, self).__init__()
            config = BertConfig.from_pretrained(pretrained_name, torchscript=True)
            config.output_attentions = False
            if num_layers:
                config.num_hidden_layers = num_layers

            self.model = BertForQuestionAnswering(config).eval()

        def forward(self, *inputs):
            input_ids = inputs[0]
            token_ids = inputs[1] if len(inputs) > 1 else None
            output = self.model(
                input_ids=input_ids, attention_mask=None, token_type_ids=token_ids
            )
            # add the span logic here
            answer_start_scores, answer_end_scores = output
            print('answer_start_scores:\n', answer_start_scores)
            print('answer_end_scores:\n', answer_end_scores)

            # Get the most likely beginning of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            # Get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  

            print('answer_start:\n', answer_start)
            print('answer_end:\n', answer_end)
                                               
            return output

    model = BertQA()
    
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
