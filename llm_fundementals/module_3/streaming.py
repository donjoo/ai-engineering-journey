import time
import anthropic

client = anthropic.Anthropic()

start = time.perf_counter()
first_token_time = None

with client.messages.stream(
    model="claude-haiku-5-5",
    max_tokens=500,
    messages=[{"role": "user", "content": "Explain recursion with a short example."}],
) as stream:
    for text in stream.text_stream:
        if first_token_time is None:
            first_token_time = time.perf_counter() - start
        print(text, end="", flush=True)      # show each piece immediately

    final = stream.get_final_message()       # the complete message, once done

total = time.perf_counter() - start
print(f"\n\nFirst text after {first_token_time:.2f}s, finished after {total:.2f}s")
print("Stop reason:", final.stop_reason)
print("Usage:", final.usage)