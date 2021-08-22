from transformers import BertTokenizer, BertForQuestionAnswering
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForQuestionAnswering.from_pretrained('bert-base-cased')
question = "What is the capital of France?"
text = "The capital of France is Paris."
inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
start, end = model(**inputs)
start_max = torch.argmax(torch.nn.functional.softmax(start, dim = -1))
end_max = torch.argmax(torch.nn.functional.softmax(end, dim = -1)) + 1 ## add one ##

# start_max = torch.argmax(start)
# end_max = torch.argmax(end) + 1
answer = tokenizer.decode(inputs["input_ids"][0][start_max : end_max])

print('answer:\n', answer)
