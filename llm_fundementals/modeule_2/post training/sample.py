from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

def run(text):
    inputs = tok(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    new_tokens = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=False)

question = "What is the capital of France?"

print("=== WITHOUT template ===")
print(run(question))

with_template = tok.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=True,   # adds "<|im_start|>assistant" so the model knows it's its turn
)
print("\n=== WITH template ===")
print(run(with_template))