from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

def generate(prompt, **settings):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt")
    out = model.generate(**inputs, tokenizer=tok, **settings)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# prompt = "Write a one-line slogan for a coffee shop."
# for temp in [0.2, 0.7, 1.5]:
#     print(f"\n=== temperature {temp} ===")
#     for _ in range(3):
#         print(generate(prompt, do_sample=True, temperature=temp, max_new_tokens=30))



import time

def make_haystack(num_lines, needle_position):
    lines = [f"Note {i}: the daily report for day {i} was routine and had no issues." for i in range(num_lines)]
    needle = "IMPORTANT: The secret code is BLUE-4471."
    index = int(needle_position * (num_lines - 1))
    lines.insert(index, needle)
    return "\n".join(lines)

for num_lines in [50, 200]:
    for position in [0.0, 0.5, 1.0]:      # start, middle, end
        haystack = make_haystack(num_lines, position)
        prompt = haystack + "\n\nWhat is the secret code? Reply with only the code."
        start = time.perf_counter()
        answer = generate(prompt, do_sample=False, max_new_tokens=15)
        seconds = time.perf_counter() - start
        print(f"lines={num_lines:>3} position={position:.1f} time={seconds:.1f}s → {answer.strip()}")