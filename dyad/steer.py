"""Activation steering hooks for reversible activation editing."""

from contextlib import contextmanager
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console

from dyad.activations import ModelUnsupportedError

console = Console()


class SteerHook:
    """Hook for applying additive steering vectors to model activations.

    Supports temporary activation editing with the formula: h' = h + αv
    where h is the original activation, v is the trait vector, and α is the steering strength.
    """

    def __init__(
        self,
        vectors: dict[int, np.ndarray],
        alpha: float,
        layers: Optional[list[int]] = None,
    ):
        """Initialize steering hook.

        Args:
            vectors: Dictionary mapping layer index to trait vector (numpy array)
            alpha: Steering strength. Recommended range: [-1.0, +1.0]. Hard-clipped to [-2.0, +2.0]
            layers: List of layer indices to apply steering. If None, uses all layers in vectors.
        """
        # Validate and clip alpha
        self.alpha = max(-2.0, min(2.0, alpha))
        if abs(alpha) > 2.0:
            console.print(
                f"[yellow]Warning: Alpha {alpha} clipped to {self.alpha}[/yellow]"
            )

        # Determine which layers to steer
        if layers is None:
            self.layers = sorted(vectors.keys())
        else:
            self.layers = sorted(set(layers))
            # Validate that all requested layers have vectors
            missing = set(self.layers) - set(vectors.keys())
            if missing:
                raise ValueError(f"Missing vectors for layers: {missing}")

        # Convert vectors to tensors and store
        self.vectors: dict[int, torch.Tensor] = {}
        self.device: Optional[torch.device] = None

        for layer_idx in self.layers:
            vec = vectors[layer_idx]
            if not isinstance(vec, torch.Tensor):
                vec = torch.from_numpy(vec).float()
            self.vectors[layer_idx] = vec

        # Store hooks
        self.hooks: list[Any] = []
        self._is_active = False

    def _get_layer_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer index."""
        # Try common model structures (same logic as ActivationLogger)
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "h"):
                return model.transformer.h[layer_idx]
            if hasattr(model.transformer, "layers"):
                return model.transformer.layers[layer_idx]
        if hasattr(model, "model"):
            if hasattr(model.model, "layers"):
                return model.model.layers[layer_idx]
            if hasattr(model.model, "h"):
                return model.model.h[layer_idx]
        if hasattr(model, "gpt_neox"):
            if hasattr(model.gpt_neox, "layers"):
                return model.gpt_neox.layers[layer_idx]

        raise ModelUnsupportedError(
            f"Could not find layer {layer_idx}. Model architecture not recognized."
        )

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""

        def hook(module: nn.Module, input: tuple, output: Any) -> Any:
            """Hook function that applies steering: h' = h + αv."""
            if not self._is_active:
                return output

            # Extract hidden states from output
            if isinstance(output, tuple):
                hidden_states = output[0]  # Shape: (batch, seq_len, hidden_dim)
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()

            if not isinstance(hidden_states, torch.Tensor):
                return output

            # Move vector to same device as hidden states
            if self.device is None:
                self.device = hidden_states.device
                # Move all vectors to device
                for k in self.vectors:
                    self.vectors[k] = self.vectors[k].to(self.device)

            vec = self.vectors[layer_idx].to(hidden_states.device)

            # Apply steering: h' = h + αv
            # vec shape: (hidden_dim,), needs to be broadcast to (batch, seq_len, hidden_dim)
            steered = hidden_states + self.alpha * vec.unsqueeze(0).unsqueeze(0)

            # Reconstruct output
            if isinstance(output, tuple):
                return (steered,) + other_outputs
            else:
                return steered

        return hook

    def attach(self, model: nn.Module) -> None:
        """Attach steering hooks to the model.

        Args:
            model: The model to attach hooks to
        """
        self.detach()  # Remove any existing hooks

        for layer_idx in self.layers:
            layer_module = self._get_layer_module(model, layer_idx)
            hook = layer_module.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def detach(self) -> None:
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._is_active = False

    def activate(self) -> None:
        """Activate steering (hooks must be attached first)."""
        if not self.hooks:
            raise RuntimeError("Hooks not attached. Call attach() first.")
        self._is_active = True

    def deactivate(self) -> None:
        """Deactivate steering (hooks remain attached but inactive)."""
        self._is_active = False

    @contextmanager
    def steering(self, model: nn.Module):
        """Context manager for temporary steering.

        Usage:
            with steer_hook.steering(model):
                # Model outputs will be steered
                output = model(input)
        """
        self.attach(model)
        self.activate()
        try:
            yield self
        finally:
            self.deactivate()
            self.detach()

    def __enter__(self):
        """Context manager entry (requires attach() to be called first)."""
        if not self.hooks:
            raise RuntimeError("Hooks not attached. Call attach() first.")
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.deactivate()
        return False


class MultiSteerHook:
    """Hook for applying multiple trait vectors simultaneously (direction mixing).

    Allows linear combination of multiple traits: h' = h + Σ(α_i * v_i)
    """

    def __init__(
        self,
        trait_vectors: dict[str, dict[int, np.ndarray]],
        alphas: dict[str, float],
        layers: Optional[list[int]] = None,
    ):
        """Initialize multi-trait steering hook.

        Args:
            trait_vectors: Dictionary mapping trait name to layer->vector dictionary
            alphas: Dictionary mapping trait name to steering strength
            layers: List of layer indices to apply steering. If None, uses intersection of all traits.
        """
        # Validate alphas
        self.alphas = {}
        for trait, alpha in alphas.items():
            clipped = max(-2.0, min(2.0, alpha))
            if abs(alpha) > 2.0:
                console.print(
                    f"[yellow]Warning: Alpha for {trait} ({alpha}) clipped to {clipped}[/yellow]"
                )
            self.alphas[trait] = clipped

        # Determine which layers to steer (intersection of all traits)
        if layers is None:
            all_layers = [set(vectors.keys()) for vectors in trait_vectors.values()]
            if not all_layers:
                raise ValueError("No trait vectors provided")
            common_layers = set.intersection(*all_layers)
            self.layers = sorted(common_layers)
        else:
            self.layers = sorted(set(layers))
            # Validate that all requested layers have vectors for all traits
            for trait, vectors in trait_vectors.items():
                missing = set(self.layers) - set(vectors.keys())
                if missing:
                    raise ValueError(
                        f"Trait '{trait}' missing vectors for layers: {missing}"
                    )

        # Combine vectors: for each layer, compute weighted sum
        self.combined_vectors: dict[int, torch.Tensor] = {}
        self.device: Optional[torch.device] = None

        for layer_idx in self.layers:
            combined = None
            for trait, vectors in trait_vectors.items():
                if layer_idx not in vectors:
                    continue
                vec = vectors[layer_idx]
                if not isinstance(vec, torch.Tensor):
                    vec = torch.from_numpy(vec).float()
                alpha = self.alphas.get(trait, 0.0)
                if combined is None:
                    combined = alpha * vec
                else:
                    combined = combined + alpha * vec
            if combined is not None:
                self.combined_vectors[layer_idx] = combined

        # Store hooks
        self.hooks: list[Any] = []
        self._is_active = False

    def _get_layer_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer index."""
        # Same logic as SteerHook
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "h"):
                return model.transformer.h[layer_idx]
            if hasattr(model.transformer, "layers"):
                return model.transformer.layers[layer_idx]
        if hasattr(model, "model"):
            if hasattr(model.model, "layers"):
                return model.model.layers[layer_idx]
            if hasattr(model.model, "h"):
                return model.model.h[layer_idx]
        if hasattr(model, "gpt_neox"):
            if hasattr(model.gpt_neox, "layers"):
                return model.gpt_neox.layers[layer_idx]

        raise ModelUnsupportedError(
            f"Could not find layer {layer_idx}. Model architecture not recognized."
        )

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""

        def hook(module: nn.Module, input: tuple, output: Any) -> Any:
            """Hook function that applies combined steering."""
            if not self._is_active:
                return output

            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()

            if not isinstance(hidden_states, torch.Tensor):
                return output

            # Move vector to same device
            if self.device is None:
                self.device = hidden_states.device
                for k in self.combined_vectors:
                    self.combined_vectors[k] = self.combined_vectors[k].to(self.device)

            vec = self.combined_vectors[layer_idx].to(hidden_states.device)

            # Apply steering: h' = h + combined_vector
            steered = hidden_states + vec.unsqueeze(0).unsqueeze(0)

            # Reconstruct output
            if isinstance(output, tuple):
                return (steered,) + other_outputs
            else:
                return steered

        return hook

    def attach(self, model: nn.Module) -> None:
        """Attach steering hooks to the model."""
        self.detach()

        for layer_idx in self.layers:
            layer_module = self._get_layer_module(model, layer_idx)
            hook = layer_module.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def detach(self) -> None:
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self._is_active = False

    def activate(self) -> None:
        """Activate steering."""
        if not self.hooks:
            raise RuntimeError("Hooks not attached. Call attach() first.")
        self._is_active = True

    def deactivate(self) -> None:
        """Deactivate steering."""
        self._is_active = False

    @contextmanager
    def steering(self, model: nn.Module):
        """Context manager for temporary steering."""
        self.attach(model)
        self.activate()
        try:
            yield self
        finally:
            self.deactivate()
            self.detach()

