"""Analysis metrics and visualization utilities."""

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from dyad.activations import ActivationLogger, load_model_and_tokenizer
from dyad.generate import load_generations
from dyad.vectors import load_vectors

console = Console()

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


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


def plot_layer_projection_curves(
    delta_proj: dict[int, float],
    output_path: Path,
    title: Optional[str] = None,
    alpha_values: Optional[dict[float, dict[int, float]]] = None,
) -> None:
    """Plot layer projection curves (Δproj vs layer index).

    Args:
        delta_proj: Dictionary mapping layer index to Δproj value
        output_path: Path to save the plot
        title: Optional plot title
        alpha_values: Optional dictionary mapping alpha to delta_proj dicts for multiple curves
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if alpha_values is None:
        # Single curve
        layers = sorted(delta_proj.keys())
        values = [delta_proj[l] for l in layers]

        ax.plot(layers, values, marker="o", linewidth=2, markersize=8, label="Δproj")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    else:
        # Multiple curves for different alpha values
        for alpha, delta_dict in sorted(alpha_values.items()):
            layers = sorted(delta_dict.keys())
            values = [delta_dict[l] for l in layers]
            ax.plot(
                layers,
                values,
                marker="o",
                linewidth=2,
                markersize=6,
                label=f"α={alpha}",
            )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Δproj (Projection Change)", fontsize=12)
    ax.set_title(title or "Layer Projection Curves", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved projection curve to {output_path}[/green]")


def plot_textual_polarity_shifts(
    generations: dict[float, str],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot textual polarity shifts (histograms and scatter plots).

    Args:
        generations: Dictionary mapping alpha to generated text
        output_path: Path to save the plot
        title: Optional plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract metrics
    alphas = sorted(generations.keys())
    lengths = [len(text.split()) for text in generations.values()]
    char_counts = [len(text) for text in generations.values()]

    # Histogram of text lengths
    ax1 = axes[0]
    ax1.hist(lengths, bins=min(10, len(lengths)), edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Text Length (words)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title("Text Length Distribution", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Scatter plot: alpha vs text length
    ax2 = axes[1]
    ax2.scatter(alphas, lengths, s=100, alpha=0.7, edgecolors="black")
    ax2.set_xlabel("Steering Strength (α)", fontsize=11)
    ax2.set_ylabel("Text Length (words)", fontsize=11)
    ax2.set_title("Steering Strength vs Text Length", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle(title or "Textual Polarity Shifts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved polarity shift plot to {output_path}[/green]")


def plot_activation_shift_heatmap(
    activations_base: dict[int, np.ndarray],
    activations_steered: dict[int, np.ndarray],
    output_path: Path,
    title: Optional[str] = None,
    max_features: int = 50,
) -> None:
    """Plot activation shift heatmap.

    Args:
        activations_base: Dictionary mapping layer index to base activations
        activations_steered: Dictionary mapping layer index to steered activations
        output_path: Path to save the plot
        title: Optional plot title
        max_features: Maximum number of features to display (for visualization)
    """
    # Get common layers
    common_layers = sorted(
        set(activations_base.keys()) & set(activations_steered.keys())
    )

    if not common_layers:
        console.print("[yellow]Warning: No common layers for heatmap[/yellow]")
        return

    # Compute mean shift per layer
    shifts = []
    for layer_idx in common_layers:
        base = activations_base[layer_idx]
        steered = activations_steered[layer_idx]

        # Mean-pool if needed
        if len(base.shape) > 1:
            base_mean = np.mean(base, axis=0)
            steered_mean = np.mean(steered, axis=0)
        else:
            base_mean = base
            steered_mean = steered

        shift = steered_mean - base_mean
        shifts.append(shift)

    # Stack into matrix: (layers, features)
    shift_matrix = np.stack(shifts)

    # Limit features for visualization
    if shift_matrix.shape[1] > max_features:
        # Sample features uniformly
        feature_indices = np.linspace(
            0, shift_matrix.shape[1] - 1, max_features, dtype=int
        )
        shift_matrix = shift_matrix[:, feature_indices]
        feature_labels = [f"F{i}" for i in feature_indices]
    else:
        feature_labels = [f"F{i}" for i in range(shift_matrix.shape[1])]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(6, len(common_layers) * 0.5)))
    im = ax.imshow(
        shift_matrix,
        aspect="auto",
        cmap="RdBu_r",
        interpolation="nearest",
    )

    # Set labels
    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(common_layers)))
    ax.set_yticklabels([f"Layer {l}" for l in common_layers], fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Activation Shift", fontsize=11)

    ax.set_xlabel("Feature Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)
    ax.set_title(
        title or "Activation Shift Heatmap", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved activation heatmap to {output_path}[/green]")


def plot_transfer_correlation(
    teacher_delta_proj: dict[int, float],
    student_delta_proj: dict[int, float],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Plot teacher-student transfer correlation scatterplot.

    Args:
        teacher_delta_proj: Dictionary mapping layer index to teacher Δproj
        student_delta_proj: Dictionary mapping layer index to student Δproj
        output_path: Path to save the plot
        title: Optional plot title
    """
    # Get common layers
    common_layers = sorted(
        set(teacher_delta_proj.keys()) & set(student_delta_proj.keys())
    )

    if len(common_layers) < 2:
        console.print("[yellow]Warning: Insufficient layers for correlation plot[/yellow]")
        return

    teacher_values = [teacher_delta_proj[l] for l in common_layers]
    student_values = [student_delta_proj[l] for l in common_layers]

    # Compute correlation
    correlation = compute_transfer_correlation(teacher_delta_proj, student_delta_proj)
    r = correlation["pearson_r"]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(teacher_values, student_values, s=100, alpha=0.7, edgecolors="black")

    # Add diagonal line
    min_val = min(min(teacher_values), min(student_values))
    max_val = max(max(teacher_values), max(student_values))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="y=x")

    # Add correlation text
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np = {correlation['pearson_p']:.3e}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Teacher Δproj", fontsize=12)
    ax.set_ylabel("Student Δproj", fontsize=12)
    ax.set_title(
        title or f"Teacher-Student Transfer (r={r:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved transfer correlation plot to {output_path}[/green]")
