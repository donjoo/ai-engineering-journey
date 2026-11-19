import math

words = {
    "pizza":  [0.9, 0.1],
    "burger": [0.8, 0.2],
    "cat":    [0.1, 0.9],
    "dog":    [0.2, 0.8],
}

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)

print("pizza vs burger:", round(cosine_similarity(words["pizza"], words["burger"]), 2))
print("pizza vs cat:   ", round(cosine_similarity(words["pizza"], words["cat"]), 2))