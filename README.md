# **Dyad**

> _Dual-view analysis of language-model personality traits through latent-space steering and transfer._

---

### **Overview**

**Dyad** is an experimental research toolkit for exploring **how stylistic or personality-like tendencies emerge and transfer** within large language models.  
It lets you discover latent "trait directions" in a model's activation space, apply them dynamically during generation, and measure how those stylistic signals persist when distilled into smaller models.

Dyad operates transparently and safely -- **no weight edits, no jailbreaks** -- using simple, interpretable activation hooks.  
The goal: build insight into how language model behavior relates to its internal geometry, while keeping safety and reproducibility at the core.

---

### **Inspiration**

This project was inspired by Rohit Krishnan's essay  
**[*Poisoned Prose - on semantic trojan horses in LLMs*](https://www.strangeloopcanon.com/p/poisoned-prose)**,  
which explores how subtle linguistic biases might propagate through model training and how understanding those mechanisms could make systems more trustworthy.

---

### **Features**

- Contrastive trait-direction discovery (trust vs. skepticism, formality vs. informality, etc.)  
- Reversible activation steering with adjustable intensity  
- Teacher-student transfer experiments using LoRA fine-tuning  
- Quantitative and visual analysis: projection shifts, polarity metrics, stability plots  
- Designed for safety and interpretability research -- all traits must be benign or stylistic  

---

### **Quickstart**

See [USAGE.md](USAGE.md) for a detailed step-by-step guide on discovering and using trait vectors.

**Basic workflow:**

```bash
# 1. Discover trait vectors from contrastive pairs
uv run dyad discover --model gpt2 --data data/formality_pairs.json --layers last-4

# 2. Apply steering during generation
uv run dyad steer --model gpt2 --trait formality --alpha 1.0 \
    --prompt "Write an email to my professor" --experiment runs/exp-<timestamp>
```

Outputs are saved to experiment directories with vectors, generations, and metadata for analysis.
