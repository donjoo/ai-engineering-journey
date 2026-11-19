from transformers import AutoTokenizer, AutoModel
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

long_text = "hello " * 600
inputs = tok(long_text, return_tensors="pt")
print("Tokens:", inputs.input_ids.shape[1])

with torch.no_grad():
    model(**inputs)   # what happens?