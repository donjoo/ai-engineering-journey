# pip install transformers torch
from transformers import AutoConfig

gpt2 = AutoConfig.from_pretrained("gpt2")
bert = AutoConfig.from_pretrained("bert-base-uncased")

print("GPT-2 (decoder-only)")
print("  layers:", gpt2.n_layer, "| vector size:", gpt2.n_embd, "| vocab:", gpt2.vocab_size)

print("BERT (encoder-only)")
print("  layers:", bert.num_hidden_layers, "| vector size:", bert.hidden_size, "| vocab:", bert.vocab_size)