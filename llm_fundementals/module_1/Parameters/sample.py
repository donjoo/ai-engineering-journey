from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

total = sum(p.numel() for p in model.parameters())
emb = model.transformer.wte.weight.numel()

print(f"Total parameters:      {total:,}")
print(f"Token embedding table: {emb:,} ({emb / total:.0%} of the model)")