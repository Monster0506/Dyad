"""Analysis metrics and visualization utilities."""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from dyad.activations import ActivationLogger, load_model_and_tokenizer
from dyad.generate import load_generations
from dyad.vectors import load_vectors

console = Console()


def compute_delta_proj(
    activations_base: dict[int, np.ndarray],
    activations_steered: dict[int, np.ndarray],
    trait_vectors: dict[int, np.ndarray],
) -> dict[int, float]:
    """Compute mean projection change (Δproj) along trait vector during steering.

    Args:
        activations_base: Dictionary mapping layer index to base activations
                         Shape: (num_texts, hidden_dim) per layer
        activations_steered: Dictionary mapping layer index to steered activations
                            Shape: (num_texts, hidden_dim) per layer
        trait_vectors: Dictionary mapping layer index to trait vector
                      Shape: (hidden_dim,) per layer

    Returns:
        Dictionary mapping layer index to mean Δproj value
    """
    delta_proj: dict[int, float] = {}

    for layer_idx in trait_vectors.keys():
        if layer_idx not in activations_base or layer_idx not in activations_steered:
            continue

        base_acts = activations_base[layer_idx]
        steered_acts = activations_steered[layer_idx]
        vec = trait_vectors[layer_idx]

        # Validate shapes
        if base_acts.shape != steered_acts.shape:
            console.print(
                f"[yellow]Warning: Shape mismatch at layer {layer_idx}, skipping[/yellow]"
            )
            continue

        # Compute projections: dot product with trait vector
        # Shape: (num_texts,)
        proj_base = np.dot(base_acts, vec)
        proj_steered = np.dot(steered_acts, vec)

        # Compute mean change
        delta = np.mean(proj_steered - proj_base)
        delta_proj[layer_idx] = float(delta)

    return delta_proj


def compute_textual_shift(
    texts_base: list[str],
    texts_steered: list[str],
    embeddings_base: Optional[dict[int, np.ndarray]] = None,
    embeddings_steered: Optional[dict[int, np.ndarray]] = None,
) -> dict[str, float]:
    """Compute textual shift metrics using embedding distances.

    Args:
        texts_base: List of base (unsteered) texts
        texts_steered: List of steered texts
        embeddings_base: Optional pre-computed embeddings for base texts
        embeddings_steered: Optional pre-computed embeddings for steered texts

    Returns:
        Dictionary with textual shift metrics:
        - 'mean_cosine_distance': Mean cosine distance between base and steered embeddings
        - 'mean_euclidean_distance': Mean Euclidean distance
    """
    if embeddings_base is None or embeddings_steered is None:
        # If embeddings not provided, compute simple bag-of-words style metrics
        # For now, return placeholder - could be enhanced with actual embeddings
        console.print(
            "[yellow]Warning: Embeddings not provided, using placeholder metrics[/yellow]"
        )
        return {
            "mean_cosine_distance": 0.0,
            "mean_euclidean_distance": 0.0,
        }

    # Compute distances
    distances_cosine = []
    distances_euclidean = []

    for layer_idx in embeddings_base.keys():
        if layer_idx not in embeddings_steered:
            continue

        base_emb = embeddings_base[layer_idx]
        steered_emb = embeddings_steered[layer_idx]

        # Mean-pool if needed (handle multiple texts)
        if len(base_emb.shape) > 1:
            base_centroid = np.mean(base_emb, axis=0)
            steered_centroid = np.mean(steered_emb, axis=0)
        else:
            base_centroid = base_emb
            steered_centroid = steered_emb

        # Cosine distance
        cos_dist = cosine(base_centroid, steered_centroid)
        distances_cosine.append(cos_dist)

        # Euclidean distance
        euc_dist = np.linalg.norm(steered_centroid - base_centroid)
        distances_euclidean.append(euc_dist)

    return {
        "mean_cosine_distance": (
            float(np.mean(distances_cosine)) if distances_cosine else 0.0
        ),
        "mean_euclidean_distance": (
            float(np.mean(distances_euclidean)) if distances_euclidean else 0.0
        ),
    }


def compute_stability_metrics(
    delta_proj_values: list[dict[int, float]],
) -> dict[str, float]:
    """Compute stability metrics across multiple runs/seeds.

    Args:
        delta_proj_values: List of Δproj dictionaries from multiple runs

    Returns:
        Dictionary with stability metrics:
        - 'mean_delta_proj': Mean Δproj across all runs
        - 'std_delta_proj': Standard deviation of Δproj
        - 'consistency': Inverse of std (higher = more consistent)
    """
    if not delta_proj_values:
        return {
            "mean_delta_proj": 0.0,
            "std_delta_proj": 0.0,
            "consistency": 0.0,
        }

    # Collect all Δproj values
    all_deltas = []
    for delta_dict in delta_proj_values:
        all_deltas.extend(delta_dict.values())

    if not all_deltas:
        return {
            "mean_delta_proj": 0.0,
            "std_delta_proj": 0.0,
            "consistency": 0.0,
        }

    mean_delta = np.mean(all_deltas)
    std_delta = np.std(all_deltas)
    consistency = 1.0 / (std_delta + 1e-10)  # Inverse of std (higher = more stable)

    return {
        "mean_delta_proj": float(mean_delta),
        "std_delta_proj": float(std_delta),
        "consistency": float(consistency),
    }


def evaluate_steering_success(
    delta_proj: dict[int, float],
    stability: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Evaluate steering success using heuristics/thresholds.

    Args:
        delta_proj: Dictionary mapping layer index to Δproj value
        stability: Optional stability metrics

    Returns:
        Dictionary with success evaluation:
        - 'mean_delta_proj': Mean Δproj across layers
        - 'max_delta_proj': Maximum Δproj
        - 'effect_strength': Categorical assessment ('mild', 'strong', 'weak')
        - 'is_successful': Boolean indicating if steering is effective
    """
    if not delta_proj:
        return {
            "mean_delta_proj": 0.0,
            "max_delta_proj": 0.0,
            "effect_strength": "none",
            "is_successful": False,
        }

    mean_delta = np.mean(list(delta_proj.values()))
    max_delta = np.max(list(delta_proj.values()))
    abs_mean_delta = abs(mean_delta)

    # Success heuristics (from spec)
    if abs_mean_delta >= 0.005:
        effect_strength = "strong"
        is_successful = True
    elif abs_mean_delta >= 0.001:
        effect_strength = "mild"
        is_successful = True
    else:
        effect_strength = "weak"
        is_successful = False

    result = {
        "mean_delta_proj": float(mean_delta),
        "max_delta_proj": float(max_delta),
        "effect_strength": effect_strength,
        "is_successful": is_successful,
    }

    # Add stability info if provided
    if stability:
        result["stability_std"] = stability.get("std_delta_proj", 0.0)
        result["stability_consistency"] = stability.get("consistency", 0.0)
        # Check consistency threshold (σ < 0.002 preferred)
        if stability.get("std_delta_proj", float("inf")) < 0.002:
            result["is_stable"] = True
        else:
            result["is_stable"] = False

    return result


def compute_transfer_correlation(
    teacher_delta_proj: dict[int, float],
    student_delta_proj: dict[int, float],
) -> dict[str, float]:
    """Compute transfer correlation between teacher and student projections.

    Args:
        teacher_delta_proj: Dictionary mapping layer index to teacher Δproj
        student_delta_proj: Dictionary mapping layer index to student Δproj

    Returns:
        Dictionary with correlation metrics:
        - 'pearson_r': Pearson correlation coefficient
        - 'pearson_p': P-value for Pearson correlation
        - 'cosine_similarity': Cosine similarity between Δproj vectors
        - 'transfer_strength': Categorical assessment ('weak', 'clear', 'strong')
    """
    # Get common layers
    common_layers = sorted(
        set(teacher_delta_proj.keys()) & set(student_delta_proj.keys())
    )

    if len(common_layers) < 2:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "cosine_similarity": 0.0,
            "transfer_strength": "none",
        }

    # Extract values in layer order
    teacher_values = [teacher_delta_proj[l] for l in common_layers]
    student_values = [student_delta_proj[l] for l in common_layers]

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(teacher_values, student_values)

    # Cosine similarity
    teacher_vec = np.array(teacher_values)
    student_vec = np.array(student_values)
    cos_sim = np.dot(teacher_vec, student_vec) / (
        np.linalg.norm(teacher_vec) * np.linalg.norm(student_vec) + 1e-10
    )

    # Transfer strength assessment (from spec)
    abs_r = abs(pearson_r)
    if abs_r >= 0.5:
        transfer_strength = "clear"
    elif abs_r >= 0.3:
        transfer_strength = "weak"
    else:
        transfer_strength = "none"

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "cosine_similarity": float(cos_sim),
        "transfer_strength": transfer_strength,
    }


def analyze_generation_pair(
    base_text: str,
    steered_text: str,
    model_name: str,
    trait_vectors_path: Path,
    alpha: float,
) -> dict[str, Any]:
    """Analyze a single generation pair (base vs steered).

    This is a simplified analysis that doesn't require re-running the model.
    For full analysis with activations, use analyze_experiment.

    Args:
        base_text: Base (unsteered) generated text
        steered_text: Steered generated text
        model_name: Model identifier
        trait_vectors_path: Path to trait vectors
        alpha: Steering strength used

    Returns:
        Dictionary with analysis results
    """
    # Load trait vectors
    try:
        trait_vectors = load_vectors(trait_vectors_path)
    except Exception as e:
        console.print(f"[red]Error loading vectors: {e}[/red]")
        return {}

    # Simple text-based metrics
    base_len = len(base_text.split())
    steered_len = len(steered_text.split())
    length_diff = steered_len - base_len

    # Character-level differences
    char_diff = len(steered_text) - len(base_text)

    return {
        "alpha": alpha,
        "base_length": base_len,
        "steered_length": steered_len,
        "length_difference": length_diff,
        "char_difference": char_diff,
        "num_layers": len(trait_vectors),
    }
