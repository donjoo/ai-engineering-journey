from transformers import AutoConfig

for name in ["gpt2", "bert-base-uncased", "Qwen/Qwen2.5-0.5B"]:
    cfg = AutoConfig.from_pretrained(name)
    print(name)
    print("  max positions:", getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None))
    print("  rope_theta:   ", getattr(cfg, "rope_theta", "not used (learned positions)"))
    print()