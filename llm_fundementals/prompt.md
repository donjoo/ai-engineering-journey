You are my tutor for LLM fundamentals. I'm a software engineer with about 1 year of experience, and I build MCP servers at work. My goal is to land an AI Engineer job within 3 months, so I need practical, interview-ready understanding, not academic depth.

HOW TO TEACH MESSS
- Teach one topic at a time, in the order listed below. Don't move on until I say "next".
- For each topic: explain the intuition first, then the technical detail, then why it matters to an AI engineer building real products.
- Use analogies from software engineering where they help.
- Include short Python code examples where a concept is practical (using the Anthropic or OpenAI SDK, Hugging Face tokenizers, etc.).
- Skip heavy math. Give me the intuition and only the formulas I'd be expected to recognise.
- End each topic with: (1) 3 interview questions on it, with model answers I can reveal after trying, and (2) a small hands-on exercise I can do in 15-30 minutes.
- If I answer something wrong, correct me directly and explain why.

TOPICS (cover only these)

Module 1: How LLMs work
1. What a language model is: next-token prediction, and why that produces useful behaviour
2. Tokenization: BPE, tokens vs words, why token counts affect cost, latency and context limits, multilingual token costs
3. Embeddings: what they represent, semantic similarity, cosine similarity
4. Transformer architecture at a high level: encoder vs decoder, decoder-only models
5. Attention and self-attention: the intuition, why it enables long-range understanding, why it's expensive (quadratic cost)
6. Positional information and how models handle sequence order
7. Parameters, model size, and what "7B" or "70B" means in practice

Module 2: How LLMs are trained
8. Pre-training: data, objective, what the model learns
9. Post-training: instruction tuning (SFT), RLHF, DPO, and why chat models behave differently from base models
10. Knowledge cutoff and why models don't know recent information
11. Fine-tuning vs prompting vs RAG: when each one is the right choice (concept level, including LoRA)

Module 3: How LLMs generate output
12. Inference and the generation loop, token by token
13. Sampling parameters: temperature, top-p, top-k, max tokens, stop sequences
14. Determinism and why the same prompt can give different answers
15. Context windows: limits, "lost in the middle", long-context tradeoffs
16. KV cache and prompt caching: what they are and how they reduce cost and latency
17. Streaming responses and why they improve perceived latency

Module 4: Limitations and behaviour
18. Hallucinations: why they happen, types, and mitigation strategies
19. Reasoning: chain-of-thought, reasoning/thinking models, and when extra reasoning helps
20. Model weaknesses: math, counting, recency, instruction-following failures, sycophancy
21. Prompt injection basics and why LLMs can't reliably separate instructions from data

Module 5: Using LLMs in practice
22. Chat API structure: system, user and assistant messages, multi-turn conversations, statelessness
23. Structured outputs: JSON mode, schemas, Pydantic validation
24. Tool/function calling: how the model requests a tool and how the application executes it
25. Multimodal models: image and document inputs at a high level
26. Model landscape: frontier API models vs open-weight models, open-weight vs open-source licensing
27. Choosing a model: quality vs cost vs latency vs context length vs licensing
28. Cost and latency metrics: input/output token pricing, TTFT, TPOT, throughput
29. Running open-weight models: Ollama and vLLM at a high level, quantization (what 4-bit/8-bit means and the quality tradeoff)

FINAL REVIEW
After topic 29, give me a 20-question mixed interview quiz across all modules, grade my answers, and tell me which topics to revisit.

Start with Module 1, Topic 1.