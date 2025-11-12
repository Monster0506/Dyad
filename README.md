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

```bash
# Create and run within a uv-managed environment
uv init dyad
uv add torch transformers datasets numpy matplotlib

# Example: discover and steer a stylistic trait
uv run python -m dyad discover --model qwen2.5b --trait formality
uv run python -m dyad steer --alpha 0.8 --prompt "Write a note of thanks to a mentor"
```

Outputs are logged with projection metrics and optional plots for inspection.
