# from https://huggingface.co/transformers/v2.8.0/usage.html#extractive-question-answering
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers.modeling_bert import BertConfig, BertForQuestionAnswering
import torch

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
# model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained(model_name)

# from transformers.modeling_bert import BertConfig
# pretrained_name = "bert-base-cased"
# config = BertConfig.from_pretrained(pretrained_name, torchscript=True)
# config.output_attentions = False
# model = BertForQuestionAnswering(config).eval()


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
    print('input_ids:\n', inputs['input_ids'])
    print('token type ids:\n', inputs['token_type_ids'])
    print('Shape of input ids:', inputs['input_ids'].shape)
    print('Shape of token ids:', inputs['token_type_ids'].shape)
    print('Shape of attn masks:', inputs['attention_mask'].shape)
    input_ids = inputs["input_ids"].tolist()[0]

    # text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # print('text tokens\n', text_tokens)

    # answer_start_scores, answer_end_scores = model(**inputs)
    answer_start_scores, answer_end_scores = model(input_ids=inputs['input_ids'],
                                                   attention_mask=None,
                                                   token_type_ids=inputs['token_type_ids'])
    print('answer start scores\n', answer_start_scores)
    print('answer end scores\n', answer_end_scores) 

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    print('answer start\n', answer_start)
    print('answer end\n', answer_end) 

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print('the texf\n',text)
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
