import random

# For each word: the possible next words and their probabilities
fake_model = {
    "The": {"cat": 0.6, "dog": 0.4},
    "cat": {"sat": 0.7, "ran": 0.3},
    "dog": {"sat": 0.5, "ran": 0.5},
    "sat": {"on": 0.8, "END": 0.2},
    "ran": {"away": 0.6, "END": 0.4},
    "on":  {"the": 0.9, "END": 0.1},
    "the": {"cat": 0.7, "mat": 0.3},
    "mat": {"END": 1.0},
    "away": {"END": 1.0},
}

def generate(mode, max_words=12):
    words = ["The"]
    for _ in range(max_words):
        options = fake_model[words[-1]]          # what can follow the last word?

        if mode == "greedy":
            next_word = max(options, key=options.get)   # always pick the most likely
        else:
            next_word = random.choices(list(options), weights=options.values())[0]  # pick randomly, weighted

        if next_word == "END":
            break
        words.append(next_word)
    return " ".join(words)

print("Greedy:", generate("greedy"))
for i in range(3):
    print("Sample:", generate("sample"))