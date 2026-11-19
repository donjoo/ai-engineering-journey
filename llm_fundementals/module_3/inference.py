import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

def time_generation(prompt, new_tokens):
    inputs = tok(prompt, return_tensors="pt")
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,      # force exactly this many tokens
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return time.perf_counter() - start, inputs.input_ids.shape[1]

time_generation("warm up", 5)   # the first call is always slow, so ignore it

for prompt, new_tokens in [("Hello", 1), ("Hello " * 500, 1), ("Hello", 100)]:
    seconds, prompt_tokens = time_generation(prompt, new_tokens)
    print(f"prompt: {prompt_tokens:>3} tokens | output: {new_tokens:>3} tokens | {seconds:.2f}s")