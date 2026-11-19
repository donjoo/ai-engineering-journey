from transformers import AutoTokenizer, AutoModel
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

def bank_vectors(sentence):
    inputs = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    bank_id = tok.convert_tokens_to_ids("bank")
    idx = inputs.input_ids[0].tolist().index(bank_id)
    # 13 vectors: index 0 = starting embeddings, 1..12 = after each block
    return [layer[0, idx] for layer in out.hidden_states]

river = bank_vectors("I sat by the river bank.")
money = bank_vectors("I put money in the bank.")

cos = torch.nn.functional.cosine_similarity
for layer, (r, m) in enumerate(zip(river, money)):
    sim = cos(r, m, dim=0).item()
    label = "embeddings" if layer == 0 else f"block {layer}"
    print(f"{label:>10}: {sim:.3f}  {'#' * round(sim * 40)}")



