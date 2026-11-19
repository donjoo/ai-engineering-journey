# pip install sentence-transformers
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What's the weather like today?",
    "I can't sign in to my account",
    "Is it going to rain?",
    "My password for the weather app",
]

embeddings = model.encode(sentences)
print(embeddings.shape)   # (3, 384): 3 sentences, 384 numbers each

scores = util.cos_sim(embeddings, embeddings)
print(scores)