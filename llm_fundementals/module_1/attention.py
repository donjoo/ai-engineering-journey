from transformers import AutoTokenizer, AutoModel
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased",
                                  output_attentions=True,
                                  attn_implementation="eager")

def show_attention(sentence, word="it"):
    inputs = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    tokens = tok.convert_ids_to_tokens(inputs.input_ids[0])
    idx = tokens.index(word)
    print(sentence)
    for layer in [2, 6, 10]:
        # average the attention from `word` across all 12 heads in this layer
        weights = out.attentions[layer][0, :, idx, :].mean(dim=0)
        top = torch.topk(weights, 5)
        print(f"  Layer {layer}:", [(tokens[i], round(w.item(), 2)) for w, i in zip(top.values, top.indices)])
    print()

show_attention("The animal didn't cross the street because it was too tired.")
show_attention("The animal didn't cross the street because it was too wide.")