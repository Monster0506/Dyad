"""Controlled text generation using activation steering."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from transformers import PreTrainedModel, PreTrainedTokenizer

from dyad.activations import load_model_and_tokenizer
from dyad.steer import SteerHook
from dyad.vectors import load_vectors

console = Console()


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    steer_hook: Optional[SteerHook] = None,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    seed: Optional[int] = None,
) -> str:
    """Generate text with optional activation steering.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        steer_hook: Optional SteerHook to apply during generation
        max_length: Maximum total length (prompt + generation)
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling (vs greedy decoding)
        seed: Random seed for reproducibility

    Returns:
        Generated text (including prompt)
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Apply steering if provided
    if steer_hook is not None:
        with steer_hook.steering(model):
            # Generate with steering
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
    else:
        # Generate without steering
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def generate_parallel(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    vectors_path: Path,
    alpha_values: list[float],
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
) -> dict[float, str]:
    """Generate parallel completions under different steering strengths.

    Args:
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt text
        vectors_path: Path to trait vectors .npz file
        alpha_values: List of steering strengths to try (e.g., [-1.0, -0.5, 0.0, 0.5, 1.0])
        max_length: Maximum total length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping alpha value to generated text
    """
    # Load trait vectors
    vectors = load_vectors(vectors_path)

    results: dict[float, str] = {}

    # Generate with base (no steering)
    console.print("[bold]Generating base (unsteered) text...[/bold]")
    base_text = generate_text(
        model,
        tokenizer,
        prompt,
        steer_hook=None,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    results[0.0] = base_text

    # Generate with each alpha value
    for alpha in alpha_values:
        if alpha == 0.0:
            continue  # Already generated
        console.print(f"[bold]Generating with α={alpha}...[/bold]")
        steer_hook = SteerHook(vectors, alpha=alpha)
        steered_text = generate_text(
            model,
            tokenizer,
            prompt,
            steer_hook=steer_hook,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        results[alpha] = steered_text

    return results


def save_generations(
    generations: dict[float, str],
    output_path: Path,
    prompt: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save generated texts with metadata.

    Args:
        generations: Dictionary mapping alpha to generated text
        output_path: Path to save JSON file
        prompt: Original prompt
        metadata: Additional metadata to include
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "prompt": prompt,
        "generations": {str(alpha): text for alpha, text in generations.items()},
        "timestamp": datetime.now().isoformat(),
    }

    if metadata:
        data["metadata"] = metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Saved generations to {output_path}[/green]")


def load_generations(path: Path) -> dict:
    """Load saved generations from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary with prompt, generations, and metadata
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert alpha strings back to floats
    if "generations" in data:
        data["generations"] = {
            float(alpha): text for alpha, text in data["generations"].items()
        }

    return data
