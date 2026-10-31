# pip install torch transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def generate(prompt, max_new_tokens=20, mode="greedy"):
    input_ids = tok(prompt, return_tensors="pt").input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]   # scores for the next token

        if mode == "greedy":
            next_id = torch.argmax(logits)              # single most likely token
        else:
            probs = torch.softmax(logits, dim=-1)       # logits -> probabilities
            next_id = torch.multinomial(probs, 1)[0]    # random draw, weighted by probability

        # stop if the model emits its end-of-text token
        if next_id.item() == tok.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)

    return tok.decode(input_ids[0])


prompt = "The capital of France is"

# Step 1: greedy
print("=== GREEDY ===")
print(generate(prompt, 20, "greedy"))

# Step 2: sampling, three runs
print("\n=== SAMPLING ===")
for i in range(3):
    print(f"--- run {i+1} ---")
    print(generate(prompt, 20, "sample"))

# Step 3: Q&A prompt on a base model
print("\n=== Q&A PROMPT (greedy, 50 tokens) ===")
print(generate("Q: What is the capital of France?\nA:", 50, "greedy"))