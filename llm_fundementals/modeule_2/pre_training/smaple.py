from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def loss_of(text):
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        return model(ids, labels=ids).loss.item()

sentences = [
    "The cat sat on the mat.",
    "Mat the on sat cat the.",
    "The capital of France is Paris.",
    "The capital of France is Berlin.",
]

for s in sentences:
    print(f"loss {loss_of(s):.2f}   {s}")