# from https://huggingface.co/transformers/v2.8.0/usage.html#extractive-question-answering

from transformers import AutoTokenizer
from transformers.modeling_bert import BertConfig, BertForQuestionAnswering
import torch

model_name = "bert-large-cased-whole-word-masking-finetuned-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]

for question in questions:
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    print(inputs['input_ids'])
    print(inputs['token_type_ids'])

    answer_start_scores, answer_end_scores = model(input_ids=inputs['input_ids'],
                                                   attention_mask=None,
                                                   token_type_ids=inputs['token_type_ids'])
    print(answer_start_scores)
    answer_start = torch.argmax(
        answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    input_ids_list = inputs["input_ids"].tolist()[0]
    print(input_ids_list)
    print(tokenizer.convert_ids_to_tokens(input_ids_list))
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids_list[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
