# pip install torch transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits   # shape: [batch, seq_len, vocab_size]

next_token_logits = logits[0, -1]       # scores for the token AFTER the last one
probs = torch.softmax(next_token_logits, dim=-1)

top = torch.topk(probs, 10)
for p, idx in zip(top.values, top.indices):
    print(f"{tok.decode(idx)!r:15} {p.item():.3f}")