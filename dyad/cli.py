"""Command-line interface for Dyad."""

import json
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from dyad.activations import (
    ActivationLogger,
    ModelUnsupportedError,
    load_model_and_tokenizer,
    parse_layer_spec,
)
from dyad.data import load_contrastive_pairs
from dyad.vectors import compute_trait_vector, save_vectors

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Dyad: Dual-view analysis of language-model persona traits."""
    pass


@cli.command()
@click.option(
    "--model",
    required=True,
    help="Model identifier (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf') or local path",
)
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to contrastive pairs JSON file",
)
@click.option(
    "--layers",
    default=None,
    help="Layer indices to use (e.g., '28,29,30,31' or 'last-4'). Default: last 4 layers",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for results. Default: runs/exp-<timestamp>",
)
@click.option(
    "--chunk-size",
    default=None,
    type=int,
    help="Process texts in chunks of this size for memory efficiency",
)
@click.option(
    "--batch-size",
    default=8,
    type=int,
    help="Batch size for processing texts",
)
@click.option(
    "--offline",
    is_flag=True,
    help="Only use local model cache, don't download from HuggingFace Hub",
)
@click.option(
    "--device",
    default=None,
    help="Device to use ('cuda', 'cpu', or None for auto)",
)
def discover(
    model: str,
    data: Path,
    layers: str | None,
    output: Path | None,
    chunk_size: int | None,
    batch_size: int,
    offline: bool,
    device: str | None,
):
    """Discover trait vectors from contrastive pairs."""
    try:
        # Create output directory
        if output is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            output = Path("runs") / f"exp-{timestamp}"
        output.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold blue]Dyad: Trait Vector Discovery[/bold blue]")
        console.print(f"Model: {model}")
        console.print(f"Dataset: {data}")
        console.print(f"Output: {output}")

        # Load model and tokenizer
        try:
            model_obj, tokenizer = load_model_and_tokenizer(
                model, device=device, offline=offline
            )
        except ModelUnsupportedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.Abort()

        # Parse layer specification
        # Create a temporary logger to get num_layers
        logger_temp = ActivationLogger(model_obj, tokenizer, layers=None)
        num_layers = logger_temp._get_num_layers()
        del logger_temp

        if layers:
            layer_list = parse_layer_spec(layers, num_layers)
        else:
            layer_list = None  # Will use default (last 4)

        console.print(f"Layers: {layer_list if layer_list else 'last-4 (default)'}")

        # Load contrastive pairs
        console.print("\n[bold]Loading dataset...[/bold]")
        positive_examples, negative_examples = load_contrastive_pairs(data)

        console.print(f"Positive examples: {len(positive_examples)}")
        console.print(f"Negative examples: {len(negative_examples)}")

        # Create activation logger
        logger = ActivationLogger(
            model_obj, tokenizer, layers=layer_list, chunk_size=chunk_size
        )

        # Extract activations for positive examples
        console.print("\n[bold]Extracting activations for positive examples...[/bold]")
        positive_activations = logger.log_activations(
            positive_examples, batch_size=batch_size
        )

        # Extract activations for negative examples
        console.print("\n[bold]Extracting activations for negative examples...[/bold]")
        negative_activations = logger.log_activations(
            negative_examples, batch_size=batch_size
        )

        # Compute trait vectors
        console.print("\n[bold]Computing trait vectors...[/bold]")
        trait_vectors = compute_trait_vector(positive_activations, negative_activations)

        # Save vectors
        vectors_dir = output / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        vectors_path = vectors_dir / "trait_vectors.npz"
        save_vectors(trait_vectors, vectors_path)

        # Save configuration
        config = {
            "model": model,
            "dataset": str(data),
            "layers": logger.layers,
            "num_positive_examples": len(positive_examples),
            "num_negative_examples": len(negative_examples),
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "timestamp": datetime.now().isoformat(),
        }
        config_path = output / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save activations (optional, for later analysis)
        activations_dir = output / "activations"
        activations_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(
            activations_dir / "positive_activations.npz", **positive_activations
        )
        np.savez_compressed(
            activations_dir / "negative_activations.npz", **negative_activations
        )

        console.print(f"\n[bold green]✓ Discovery complete![/bold green]")
        console.print(f"Trait vectors saved to: {vectors_path}")
        console.print(f"Configuration saved to: {config_path}")
        console.print(f"Experiment directory: {output}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise click.Abort()


@cli.command()
def steer():
    """Apply activation steering on prompts."""
    console.print("[yellow]Steer command not yet implemented[/yellow]")
    pass


@cli.command()
def analyze():
    """Generate quantitative and visual reports."""
    console.print("[yellow]Analyze command not yet implemented[/yellow]")
    pass


@cli.command()
def student():
    """Train student model with LoRA fine-tuning."""
    console.print("[yellow]Student command not yet implemented[/yellow]")
    pass


@cli.group()
def utils():
    """Utility commands."""
    pass


@utils.command()
def validate():
    """Validate dataset JSON schema."""
    console.print("[yellow]Validate command not yet implemented[/yellow]")
    pass


@utils.command()
def list_traits():
    """Show safe traits whitelist."""
    console.print("[yellow]List-traits command not yet implemented[/yellow]")
    pass


@utils.command()
def clean():
    """Clean up runs directory."""
    console.print("[yellow]Clean command not yet implemented[/yellow]")
    pass


@utils.command()
def seed():
    """Set random seed utilities."""
    console.print("[yellow]Seed command not yet implemented[/yellow]")
    pass


if __name__ == "__main__":
    cli()

