"""Activation logging and extraction utilities."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


console = Console()


class ModelUnsupportedError(Exception):
    """Raised when model architecture is not supported."""

    pass


class ActivationLogger:
    """Logs activations from specified transformer layers with memory-efficient streaming."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        layers: Optional[list[int]] = None,
        chunk_size: Optional[int] = None,
    ):
        """Initialize activation logger.

        Args:
            model: Pre-trained transformer model
            tokenizer: Tokenizer for the model
            layers: List of layer indices to log. If None, uses last 4 layers.
            chunk_size: Process texts in chunks of this size. If None, processes all at once.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.chunk_size = chunk_size

        # Determine which layers to log
        num_layers = self._get_num_layers()
        if layers is None:
            # Default to last 4 layers
            self.layers = list(range(max(0, num_layers - 4), num_layers))
        else:
            self.layers = sorted(set(layers))
            # Validate layer indices
            if any(l < 0 or l >= num_layers for l in self.layers):
                raise ValueError(
                    f"Invalid layer indices. Model has {num_layers} layers, got {self.layers}"
                )

        # Storage for activations (mean-pooled per layer)
        self.activations: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.hooks: list[Any] = []

        # Track if we're currently logging
        self._is_logging = False

    def _get_num_layers(self) -> int:
        """Get the number of transformer layers in the model."""
        # Try common attribute names
        if hasattr(self.model, "config"):
            config = self.model.config
            # Common config attributes
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
            if hasattr(config, "n_layer"):
                return config.n_layer
            if hasattr(config, "num_layers"):
                return config.num_layers

        # Try to find transformer blocks directly
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return len(self.model.transformer.h)
            if hasattr(self.model.transformer, "layers"):
                return len(self.model.transformer.layers)
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return len(self.model.model.layers)
            if hasattr(self.model.model, "h"):
                return len(self.model.model.h)

        raise ModelUnsupportedError(
            "Could not determine number of layers. Model architecture not recognized."
        )

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer index."""
        # Try common model structures
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h[layer_idx]
            if hasattr(self.model.transformer, "layers"):
                return self.model.transformer.layers[layer_idx]
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers[layer_idx]
            if hasattr(self.model.model, "h"):
                return self.model.model.h[layer_idx]
        if hasattr(self.model, "gpt_neox"):
            if hasattr(self.model.gpt_neox, "layers"):
                return self.model.gpt_neox.layers[layer_idx]

        raise ModelUnsupportedError(
            f"Could not find layer {layer_idx}. Model architecture not recognized."
        )

    def _hook_fn(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""

        def hook(module: nn.Module, input: tuple, output: Any) -> None:
            """Hook function that extracts and mean-pools activations."""
            if not self._is_logging:
                return

            # Extract hidden states from output
            # Output is typically a tuple, with hidden states as first element
            if isinstance(output, tuple):
                hidden_states = output[0]  # Shape: (batch, seq_len, hidden_dim)
            else:
                hidden_states = output

            if not isinstance(hidden_states, torch.Tensor):
                return

            # Mean-pool over sequence dimension (excluding padding)
            # Assume padding tokens have value 0 or are masked
            # For now, we'll mean-pool all tokens (can be improved with attention masks)
            mean_pooled = hidden_states.mean(dim=1)  # Shape: (batch, hidden_dim)

            # Store activation (detach to avoid gradient tracking)
            self.activations[layer_idx].append(mean_pooled.detach().cpu())

        return hook

    def _attach_hooks(self) -> None:
        """Attach forward hooks to specified layers."""
        self._remove_hooks()  # Clean up any existing hooks

        for layer_idx in self.layers:
            layer_module = self._get_layer_module(layer_idx)
            hook = layer_module.register_forward_hook(self._hook_fn(layer_idx))
            self.hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def log_activations(
        self, texts: list[str], batch_size: int = 8, show_progress: bool = True
    ) -> dict[int, np.ndarray]:
        """Log activations for a list of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping layer index to numpy array of activations
            Shape: (num_texts, hidden_dim) per layer
        """
        # Clear previous activations
        self.activations.clear()
        self._attach_hooks()
        self._is_logging = True

        try:
            # Process in chunks if specified
            if self.chunk_size:
                text_chunks = [
                    texts[i : i + self.chunk_size]
                    for i in range(0, len(texts), self.chunk_size)
                ]
            else:
                text_chunks = [texts]

            iterator = text_chunks
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(text_chunks, desc="Processing texts")

            for chunk in iterator:
                # Tokenize chunk
                encoded = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                # Forward pass (activations will be captured by hooks)
                try:
                    with torch.no_grad():
                        self.model(**encoded, output_hidden_states=False)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to process batch: {e}[/yellow]"
                    )
                    continue

        finally:
            self._is_logging = False
            self._remove_hooks()

        # Concatenate all activations per layer
        result: dict[int, np.ndarray] = {}
        for layer_idx in self.layers:
            if self.activations[layer_idx]:
                # Stack all batch activations: (total_texts, hidden_dim)
                stacked = torch.cat(self.activations[layer_idx], dim=0)
                result[layer_idx] = stacked.numpy()
            else:
                console.print(
                    f"[yellow]Warning: No activations captured for layer {layer_idx}[/yellow]"
                )
                result[layer_idx] = np.array([])

        return result

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self._remove_hooks()
        return False


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    """Parse layer specification string into list of layer indices.

    Args:
        spec: Layer specification (e.g., "28,29,30,31" or "last-4")
        num_layers: Total number of layers in the model

    Returns:
        List of layer indices

    Examples:
        >>> parse_layer_spec("28,29,30,31", 32)
        [28, 29, 30, 31]
        >>> parse_layer_spec("last-4", 32)
        [28, 29, 30, 31]
    """
    spec = spec.strip().lower()

    if spec.startswith("last-"):
        n = int(spec.split("-")[1])
        return list(range(max(0, num_layers - n), num_layers))

    # Parse comma-separated list
    try:
        layers = [int(x.strip()) for x in spec.split(",")]
        return layers
    except ValueError:
        raise ValueError(f"Invalid layer specification: {spec}")


def load_model_and_tokenizer(
    model_name: str, device: Optional[str] = None, offline: bool = False
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load model and tokenizer from HuggingFace Hub or local path.

    Args:
        model_name: Model identifier or local path
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        offline: If True, only use local cache

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ModelUnsupportedError: If model architecture is not supported
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[blue]Loading model: {model_name}[/blue]")
    console.print(f"[blue]Device: {device}[/blue]")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=offline,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            device_map="auto" if device == "cuda" else None,
        )

        # Move to device if not using device_map
        if device != "cuda" or not hasattr(model, "hf_device_map"):
            model = model.to(device)

        model.eval()  # Set to evaluation mode

        console.print("[green]Model loaded successfully[/green]")
        return model, tokenizer

    except Exception as e:
        if "offline" in str(e).lower() or "local_files_only" in str(e).lower():
            raise ModelUnsupportedError(
                f"Model {model_name} not found in local cache. Set offline=False to download."
            ) from e
        raise ModelUnsupportedError(f"Failed to load model {model_name}: {e}") from e
