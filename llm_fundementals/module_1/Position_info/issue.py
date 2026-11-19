import numpy as np

np.random.seed(0)
words = ["dog", "bites", "man"]
emb = {w: np.random.randn(4) for w in words}   # fake 4-number embeddings

# def attend(query_word, sentence):
#     q = emb[query_word]
#     K = np.array([emb[w] for w in sentence])
#     scores = K @ q / np.sqrt(4)
#     weights = np.exp(scores) / np.exp(scores).sum()
#     return weights @ K

positions = [np.random.randn(4) for _ in range(3)]   # one vector per position: 0, 1, 2

def attend_with_position(query_word, sentence):
    vecs = [emb[w] + positions[i] for i, w in enumerate(sentence)]   # word + position
    q = vecs[sentence.index(query_word)]
    K = np.array(vecs)
    scores = K @ q / np.sqrt(4)
    weights = np.exp(scores) / np.exp(scores).sum()
    return weights @ K

print("dog bites man:", attend_with_position("bites", ["dog", "bites", "man"]).round(3))
print("man bites dog:", attend_with_position("bites", ["man", "bites", "dog"]).round(3))
# print("dog bites man:", attend("bites", ["dog", "bites", "man"]).round(3))
# print("man bites dog:", attend("bites", ["man", "bites", "dog"]).round(3))