# pip install tiktoken
import tiktoken

# enc = tiktoken.get_encoding("cl100k_base")   # a tokenizer used by OpenAI models
enc = tiktoken.get_encoding("o200k_base")

# text = "I love unbelievable pizza!"
english = "the weather is very nice today ,so let's go for a walk."
hindi = "आज मौसम बहुत अच्छा है, तो चलो टहलने चलते हैं।"
dic = ["hello", " hello", "Hello", "HELLO","antidisestablishmentarianism","def getUserById(user_id):",english,hindi]


for text in dic:
    ids = enc.encode(text)

    print("Token IDs:", ids)
    print("Number of tokens:", len(ids))
    print("Pieces:", [enc.decode([i]) for i in ids])




en_tokens = len(enc.encode(english))
hi_tokens = len(enc.encode(hindi))

print("English tokens:", en_tokens)
print("Hindi tokens:  ", hi_tokens)

ratio = hi_tokens / en_tokens
print(f"Hindi uses {ratio:.1f}x as many tokens as English")









def daily_cost(input_tokens, output_tokens, requests,
               input_price_per_m=3, output_price_per_m=15):
    input_cost = input_tokens * requests / 1_000_000 * input_price_per_m
    output_cost = output_tokens * requests / 1_000_000 * output_price_per_m
    return input_cost + output_cost

ratio = 3.0   

english = daily_cost(2000, 500, 10_000)
hindi = daily_cost(2000 * ratio, 500 * ratio, 10_000)

print(f"English: ${english:.2f}/day, ${english * 30:,.2f}/month")
print(f"Hindi:   ${hindi:.2f}/day, ${hindi * 30:,.2f}/month")



dumm= "In a real app, not everything gets bigger. Your system prompt (the fixed instructions) is often written in English no matter what language the user speaks. Only the user’s message and the model’s reply are in Hindi. So the real increase is usually somewhat smaller than the full ratio. Saying “I’d measure which parts of the prompt are actually in the user’s language”."