from transformers import pipeline

prompt = "What is the capital of France?"

base = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B")
print("=== BASE ===")
print(base(prompt, max_new_tokens=60)[0]["generated_text"])

chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
print("\n=== INSTRUCT ===")
result = chat([{"role": "user", "content": prompt}], max_new_tokens=60)
print(result[0]["generated_text"][-1]["content"])