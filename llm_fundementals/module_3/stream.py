import time
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

text = tok.apply_chat_template(
    [{"role": "user", "content": "Explain recursion with a short Python example."}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tok(text, return_tensors="pt")

# 1. Without streaming: nothing appears until the end
start = time.perf_counter()
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
print(f"\n[Non-streaming] saw everything after {time.perf_counter() - start:.2f}s\n")

# 2. With streaming: text appears as it's generated
streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
start = time.perf_counter()
thread = Thread(target=model.generate,
                kwargs=dict(**inputs, max_new_tokens=200, do_sample=False, streamer=streamer))
thread.start()

first = None
for chunk in streamer:
    if first is None and chunk:
        first = time.perf_counter() - start
    print(chunk, end="", flush=True)
thread.join()
print(f"\n\n[Streaming] first text after {first:.2f}s, finished after {time.perf_counter() - start:.2f}s")