# https://huggingface.co/transformers/v2.8.0/usage.html#extractive-question-answering
# other helpful place
# https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/huggingface_pytorch-transformers.ipynb#scrollTo=vGOLOM1iRIbN

import torch
from torch import nn
from transformers.modeling_bert import BertForMaskedLM
from transformers import AutoTokenizer

model_name = "bert-base-cased"

class UnmaskWithMLM(nn.Module):
    def __init__(self):
        super(UnmaskWithMLM, self).__init__()
        pretrained_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)                
        self.model = BertForMaskedLM.from_pretrained(pretrained_name)


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
logits = model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

inputs = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]

token_logits = model(input)
print(token_logits)
print(token_logits.shape)


mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
