import re
import anthropic

client = anthropic.Anthropic()

SYSTEM = """Answer the user's question using ONLY the documents provided.
First, copy the exact sentences from the documents that support your answer,
one per line, inside <quotes></quotes>.
Then give your answer inside <answer></answer>.
If the documents don't contain the answer, reply exactly:
<answer>I don't know based on the provided documents.</answer>"""

documents = [
    "Refunds are available within 30 days of purchase.",
    "Refunds are issued to the original payment method within 5 business days.",
]

def ask(question):
    docs_text = "\n".join(f"<document>{d}</document>" for d in documents)
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=1024,                     # thinking tokens count toward this, so leave room
        output_config={"effort": "low"},     # temperature was removed; effort is the knob now
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{docs_text}\n\nQuestion: {question}"}],
    )
    # content can start with a thinking block, so grab the text block explicitly
    return next(b.text for b in response.content if b.type == "text")

def quotes_are_real(response_text):
    """Check that every quote the model gave actually appears in the documents."""
    match = re.search(r"<quotes>(.*?)</quotes>", response_text, re.DOTALL)
    if not match:
        return True   # no quotes given (for example, an "I don't know" answer)
    quotes = [q.strip().strip('"') for q in match.group(1).split("\n") if q.strip()]
    source = " ".join(documents)
    return all(q in source for q in quotes)

result = ask("How long do I have to request a refund?")
print(result)
print("Quotes verified:", quotes_are_real(result))