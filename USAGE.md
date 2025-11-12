# Dyad Quick Start Guide: Formality Trait

This guide walks you through discovering and using a formality trait vector with Dyad.

## Prerequisites

- Python ≥3.11
- `uv` package manager installed
- Dependencies installed: `uv sync`

## Step 1: Prepare Your Dataset

Create a JSON file with contrastive pairs showing formal vs informal examples.

**Example: `data/formality_pairs.json`**

```json
{
  "trait": "formality",
  "positive_label": "formal",
  "negative_label": "informal",
  "pairs": [
    {
      "formal": "Dear Dr. Smith, I would like to request a meeting to discuss the project proposal.",
      "informal": "Hey, can we chat about the project?"
    },
    {
      "formal": "I am writing to confirm our meeting scheduled for tomorrow at 2:00 PM.",
      "informal": "Just checking if we're still on for tomorrow at 2?"
    },
    {
      "formal": "I would be grateful if you could provide me with the requested information.",
      "informal": "Could you send me that info? Thanks!"
    }
  ]
}
```

**Dataset Format:**
- `trait`: Name of the trait (e.g., "formality")
- `positive_label`: Label for the positive direction (e.g., "formal")
- `negative_label`: Label for the negative direction (e.g., "informal")
- `pairs`: List of dictionaries, each containing both labels as keys

**Tip:** Include at least 10-20 pairs for better results. More examples = stronger trait vectors.

## Step 2: Discover Trait Vectors

Run the `discover` command to extract activations and compute trait vectors:

```bash
uv run dyad discover \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data data/formality_pairs.json \
    --layers last-4 \
    --batch-size 8 \
    --output runs/formality-exp
```

**Parameters:**
- `--model`: Model identifier (e.g., `gpt2`, `gpt2-medium`, `Qwen/Qwen2.5-7B-Instruct`)
- `--data`: Path to your contrastive pairs JSON file
- `--layers`: Which layers to use (e.g., `last-4`, `28,29,30,31`, or omit for default last-4)
- `--batch-size`: Batch size for processing (adjust based on memory)
- `--output`: Where to save results (default: `runs/exp-<timestamp>`)

**What this does:**
1. Loads the model and tokenizer
2. Extracts activations from positive (formal) examples
3. Extracts activations from negative (informal) examples
4. Computes trait vectors as normalized differences
5. Saves vectors, activations, and configuration to the output directory

**Output:**
- `runs/formality-exp/vectors/trait_vectors.npz` - Trait vectors (one per layer)
- `runs/formality-exp/config.json` - Experiment configuration
- `runs/formality-exp/activations/` - Saved activations for analysis

## Step 3: Apply Steering

Use the `steer` command to generate text with activation steering:

### Single Generation

```bash
uv run dyad steer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait formality \
    --alpha 1.0 \
    --prompt "Write an email to my professor" \
    --experiment runs/formality-exp \
    --max-length 100 \
    --temperature 0.7
```

**Parameters:**
- `--model`: Same model used in discover (or compatible model)
- `--trait`: Trait name (for reference, doesn't affect computation)
- `--alpha`: Steering strength
  - Positive values (e.g., `1.0`, `1.5`) → more formal
  - Negative values (e.g., `-1.0`, `-1.5`) → more informal
  - `0.0` → no steering (baseline)
- `--prompt`: Input text to complete
- `--experiment`: Path to experiment directory from discover
- `--max-length`: Maximum generation length
- `--temperature`: Sampling temperature (0.0 = deterministic, higher = more random)

### Parallel Generation (Compare Multiple Alpha Values)

Generate multiple versions with different steering strengths:

```bash
uv run dyad steer \
    --model gpt2 \
    --trait formality \
    --alpha 0.0 \
    --prompt "Can you help me with this?" \
    --experiment runs/formality-exp \
    --parallel \
    --alpha-range "-1.5,0.0,1.5" \
    --max-length 80 \
    --seed 42
```

**Additional Parameters:**
- `--parallel`: Enable parallel generation mode
- `--alpha-range`: Comma-separated alpha values to try (e.g., `-1.5,0.0,1.5`)
- `--seed`: Random seed for reproducibility

**Output:**
- Generations saved to `runs/formality-exp/generations/generation_<timestamp>.json`
- Contains all generated texts with metadata

## Example Workflow

```bash
# 1. Discover trait vectors
uv run dyad discover \
    --model gpt2 \
    --data data/formality_pairs.json \
    --layers last-4 \
    --output runs/formality-exp

# 2. Generate with formal steering (α=1.5)
uv run dyad steer \
    --model gpt2 \
    --trait formality \
    --alpha 1.5 \
    --prompt "I need to ask you something" \
    --experiment runs/formality-exp \
    --max-length 60

# 3. Generate with informal steering (α=-1.5)
uv run dyad steer \
    --model gpt2 \
    --trait formality \
    --alpha -1.5 \
    --prompt "I need to ask you something" \
    --experiment runs/formality-exp \
    --max-length 60

# 4. Compare multiple steering strengths
uv run dyad steer \
    --model gpt2 \
    --trait formality \
    --alpha 0.0 \
    --prompt "I need to ask you something" \
    --experiment runs/formality-exp \
    --parallel \
    --alpha-range "-2.0,-1.0,0.0,1.0,2.0" \
    --max-length 60 \
    --seed 42
```

## Tips for Better Results

1. **More Training Data**: Use 20-50+ contrastive pairs for stronger trait vectors
2. **Larger Models**: Try `gpt2-medium`, `gpt2-large`, or `Qwen/Qwen2.5-7B-Instruct` for better quality
3. **More Layers**: Steer more layers (e.g., `last-6` or `last-8`) for stronger effects
4. **Higher Alpha**: Use α values in range [-2.0, 2.0] for more noticeable effects (but may reduce quality)
5. **Reproducibility**: Use `--seed` for consistent comparisons
6. **GPU**: Use `--device cuda` if you have a GPU for faster processing

## Troubleshooting

**"No experiments found"**
- Make sure you've run `discover` first
- Check that the experiment directory exists

**"Trait vectors not found"**
- Verify the experiment path is correct
- Check that `vectors/trait_vectors.npz` exists in the experiment directory

**Weak steering effects**
- Try more training examples
- Use a larger model
- Steer more layers
- Increase alpha values (but watch for quality degradation)

**Out of memory**
- Reduce `--batch-size` in discover
- Use a smaller model
- Process in chunks with `--chunk-size`

## Next Steps

- Try other traits (verbosity, curiosity, etc.)
- Experiment with different models
- Use `dyad analyze` (coming in Phase III) to quantify steering effects
- Fine-tune a student model with `dyad student` (coming in Phase IV)


