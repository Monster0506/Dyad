"""Trait vector computation and normalization."""

from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console

console = Console()


def compute_trait_vector(
    positive_activations: dict[int, np.ndarray],
    negative_activations: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Compute trait vector as normalized difference between positive and negative activations.

    Args:
        positive_activations: Dictionary mapping layer index to activation array
                             Shape: (num_examples, hidden_dim) per layer
        negative_activations: Dictionary mapping layer index to activation array
                             Shape: (num_examples, hidden_dim) per layer

    Returns:
        Dictionary mapping layer index to normalized trait vector
        Shape: (hidden_dim,) per layer

    Raises:
        ValueError: If layer sets don't match or shapes are incompatible
    """
    # Validate that both have the same layers
    pos_layers = set(positive_activations.keys())
    neg_layers = set(negative_activations.keys())
    if pos_layers != neg_layers:
        raise ValueError(
            f"Layer mismatch: positive has {pos_layers}, negative has {neg_layers}"
        )

    trait_vectors: dict[int, np.ndarray] = {}

    for layer_idx in pos_layers:
        pos_acts = positive_activations[layer_idx]
        neg_acts = negative_activations[layer_idx]

        # Validate shapes
        if pos_acts.shape != neg_acts.shape:
            raise ValueError(
                f"Shape mismatch at layer {layer_idx}: "
                f"positive {pos_acts.shape} vs negative {neg_acts.shape}"
            )

        # Compute mean activations
        mean_pos = np.mean(pos_acts, axis=0)  # Shape: (hidden_dim,)
        mean_neg = np.mean(neg_acts, axis=0)  # Shape: (hidden_dim,)

        # Compute directional vector
        v_trait = mean_pos - mean_neg  # Shape: (hidden_dim,)

        # L2 normalization to unit length
        norm = np.linalg.norm(v_trait)
        if norm > 1e-10:  # Avoid division by zero
            v_trait = v_trait / norm
        else:
            console.print(
                f"[yellow]Warning: Trait vector at layer {layer_idx} has near-zero norm, using unnormalized vector[/yellow]"
            )

        trait_vectors[layer_idx] = v_trait

        console.print(
            f"[green]Computed trait vector for layer {layer_idx}: "
            f"shape={v_trait.shape}, norm={np.linalg.norm(v_trait):.6f}[/green]"
        )

    return trait_vectors


def save_vectors(vectors: dict[int, np.ndarray], path: Path) -> None:
    """Save trait vectors to NumPy .npz file.

    Args:
        vectors: Dictionary mapping layer index to trait vector
        path: Path where to save the .npz file
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict with string keys (npz requires this)
    save_dict = {f"layer_{layer_idx}": vec for layer_idx, vec in vectors.items()}
    save_dict["layer_indices"] = np.array(list(vectors.keys()))

    np.savez_compressed(path, **save_dict)
    console.print(f"[green]Saved trait vectors to {path}[/green]")


def load_vectors(path: Path) -> dict[int, np.ndarray]:
    """Load trait vectors from NumPy .npz file.

    Args:
        path: Path to the .npz file

    Returns:
        Dictionary mapping layer index to trait vector

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    data = np.load(path, allow_pickle=False)

    # Extract layer indices
    if "layer_indices" in data:
        layer_indices = data["layer_indices"].tolist()
    else:
        # Fallback: infer from keys
        layer_indices = [
            int(key.split("_")[1])
            for key in data.keys()
            if key.startswith("layer_") and key != "layer_indices"
        ]

    # Reconstruct dictionary
    vectors: dict[int, np.ndarray] = {}
    for layer_idx in layer_indices:
        key = f"layer_{layer_idx}"
        if key in data:
            vectors[layer_idx] = data[key]
        else:
            raise ValueError(f"Missing vector for layer {layer_idx} in {path}")

    console.print(f"[green]Loaded {len(vectors)} trait vectors from {path}[/green]")
    return vectors


def compute_layer_stability(
    vectors: dict[int, np.ndarray], reference_layer: Optional[int] = None
) -> dict[str, float]:
    """Compute stability metrics for trait vectors across layers.

    Args:
        vectors: Dictionary mapping layer index to trait vector
        reference_layer: Layer to use as reference for cosine similarity.
                        If None, uses the first layer.

    Returns:
        Dictionary with stability metrics:
        - 'mean_cosine_similarity': Mean cosine similarity between adjacent layers
        - 'min_cosine_similarity': Minimum cosine similarity between any two layers
        - 'max_cosine_similarity': Maximum cosine similarity between any two layers
        - 'std_cosine_similarity': Standard deviation of cosine similarities
    """
    if len(vectors) < 2:
        return {
            "mean_cosine_similarity": 1.0,
            "min_cosine_similarity": 1.0,
            "max_cosine_similarity": 1.0,
            "std_cosine_similarity": 0.0,
        }

    layers = sorted(vectors.keys())
    similarities = []

    # Compute cosine similarity between all pairs
    for i, layer_i in enumerate(layers):
        vec_i = vectors[layer_i]
        for layer_j in layers[i + 1 :]:
            vec_j = vectors[layer_j]
            # Cosine similarity
            cos_sim = np.dot(vec_i, vec_j) / (
                np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
            )
            similarities.append(cos_sim)

    similarities = np.array(similarities)

    return {
        "mean_cosine_similarity": float(np.mean(similarities)),
        "min_cosine_similarity": float(np.min(similarities)),
        "max_cosine_similarity": float(np.max(similarities)),
        "std_cosine_similarity": float(np.std(similarities)),
    }


def validate_vector_shapes(
    vectors: dict[int, np.ndarray], expected_hidden_dim: Optional[int] = None
) -> bool:
    """Validate that all vectors have consistent shapes.

    Args:
        vectors: Dictionary mapping layer index to trait vector
        expected_hidden_dim: Expected hidden dimension. If None, infers from first vector.

    Returns:
        True if all shapes are consistent

    Raises:
        ValueError: If shapes are inconsistent
    """
    if not vectors:
        raise ValueError("Empty vector dictionary")

    layers = sorted(vectors.keys())
    first_shape = vectors[layers[0]].shape

    if len(first_shape) != 1:
        raise ValueError(f"Expected 1D vector, got shape {first_shape}")

    hidden_dim = first_shape[0]
    if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"Hidden dimension mismatch: expected {expected_hidden_dim}, got {hidden_dim}"
        )

    for layer_idx in layers[1:]:
        vec_shape = vectors[layer_idx].shape
        if vec_shape != first_shape:
            raise ValueError(
                f"Shape mismatch: layer {layers[0]} has {first_shape}, "
                f"layer {layer_idx} has {vec_shape}"
            )

    return True

