import anthropic

client = anthropic.Anthropic()

handbook_text = open("handbook.txt").read()   # a long document, reused across requests

def ask(question):
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=300,
        system=[
            {"type": "text", "text": "Answer questions using only the employee handbook below."},
            {
                "type": "text",
                "text": handbook_text,
                "cache_control": {"type": "ephemeral"},   # cache everything up to here
            },
        ],
        messages=[{"role": "user", "content": question}],   # changes every time, so it goes last
    )
    print(response.usage)   # shows cache_creation_input_tokens and cache_read_input_tokens
    return response.content[0].text

ask("How many vacation days do I get?")      # first call: writes the cache
ask("What is the remote work policy?")       # second call: reads from the cache