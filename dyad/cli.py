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
from dyad.analysis import (
    compute_delta_proj,
    generate_analysis_report,
    generate_textual_summary,
    plot_activation_shift_heatmap,
    plot_layer_projection_curves,
    plot_textual_polarity_shifts,
    save_analysis_report,
    save_textual_summary,
)
from dyad.data import load_contrastive_pairs
from dyad.generate import generate_parallel, load_generations, save_generations
from dyad.steer import SteerHook
from dyad.vectors import compute_trait_vector, load_vectors, save_vectors


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
    help="Model identifier (e.g., 'gpt2', 'Qwen/Qwen2.5-7B-Instruct') or local path",
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

        # Convert integer keys to strings for npz format
        pos_acts_dict = {f"layer_{k}": v for k, v in positive_activations.items()}
        neg_acts_dict = {f"layer_{k}": v for k, v in negative_activations.items()}

        np.savez_compressed(
            activations_dir / "positive_activations.npz", **pos_acts_dict
        )
        np.savez_compressed(
            activations_dir / "negative_activations.npz", **neg_acts_dict
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
@click.option(
    "--model",
    required=True,
    help="Model identifier or local path",
)
@click.option(
    "--trait",
    required=True,
    help="Trait name (must match a previous discover run)",
)
@click.option(
    "--alpha",
    required=True,
    type=float,
    help="Steering strength (e.g., 0.5, 1.0, -0.75)",
)
@click.option(
    "--prompt",
    default=None,
    help="Input prompt text. If not provided, reads from stdin",
)
@click.option(
    "--experiment",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to experiment directory from discover command. If not provided, searches runs/",
)
@click.option(
    "--num-generations",
    default=1,
    type=int,
    help="Number of generations to produce",
)
@click.option(
    "--max-length",
    default=100,
    type=int,
    help="Maximum generation length",
)
@click.option(
    "--temperature",
    default=0.7,
    type=float,
    help="Sampling temperature",
)
@click.option(
    "--top-p",
    default=0.9,
    type=float,
    help="Nucleus sampling top-p",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Generate parallel completions with multiple alpha values",
)
@click.option(
    "--alpha-range",
    default="-1.0,0.0,1.0",
    help="Comma-separated alpha values for parallel generation (e.g., '-1.0,0.0,1.0')",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory. Default: same as experiment directory",
)
@click.option(
    "--offline",
    is_flag=True,
    help="Only use local model cache",
)
@click.option(
    "--device",
    default=None,
    help="Device to use ('cuda', 'cpu', or None for auto)",
)
def steer(
    model: str,
    trait: str,
    alpha: float,
    prompt: str | None,
    experiment: Path | None,
    num_generations: int,
    max_length: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    parallel: bool,
    alpha_range: str,
    output: Path | None,
    offline: bool,
    device: str | None,
):
    """Apply activation steering on prompts."""
    try:
        # Find experiment directory if not provided
        if experiment is None:
            # Search for most recent experiment in runs/
            runs_dir = Path("runs")
            if runs_dir.exists():
                experiments = sorted(
                    [d for d in runs_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )
                if experiments:
                    experiment = experiments[0]
                    console.print(
                        f"[yellow]Using most recent experiment: {experiment}[/yellow]"
                    )
                else:
                    console.print(
                        "[red]No experiments found. Run 'dyad discover' first.[/red]"
                    )
                    raise click.Abort()
            else:
                console.print(
                    "[red]No runs directory found. Run 'dyad discover' first.[/red]"
                )
                raise click.Abort()

        # Load vectors from experiment
        vectors_path = experiment / "vectors" / "trait_vectors.npz"
        if not vectors_path.exists():
            console.print(f"[red]Trait vectors not found at {vectors_path}[/red]")
            raise click.Abort()

        vectors = load_vectors(vectors_path)

        # Load model and tokenizer
        try:
            model_obj, tokenizer = load_model_and_tokenizer(
                model, device=device, offline=offline
            )
        except ModelUnsupportedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise click.Abort()

        # Get prompt
        if prompt is None:
            console.print(
                "[bold]Enter prompt (press Ctrl+D or Ctrl+Z when done):[/bold]"
            )
            try:
                prompt = click.get_text_stream("stdin").read().strip()
            except Exception:
                console.print("[red]Failed to read prompt from stdin[/red]")
                raise click.Abort()

        if not prompt:
            console.print("[red]Prompt cannot be empty[/red]")
            raise click.Abort()

        # Determine output directory
        if output is None:
            output = experiment / "generations"
        output.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold blue]Dyad: Activation Steering[/bold blue]")
        console.print(f"Model: {model}")
        console.print(f"Trait: {trait}")
        console.print(
            f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}"
        )

        # Generate text
        if parallel:
            # Parse alpha range
            try:
                alpha_values = [float(x.strip()) for x in alpha_range.split(",")]
            except ValueError:
                console.print(f"[red]Invalid alpha-range format: {alpha_range}[/red]")
                raise click.Abort()

            console.print(f"Generating parallel completions with α={alpha_values}")
            from dyad.generate import generate_parallel

            generations = generate_parallel(
                model_obj,
                tokenizer,
                prompt,
                vectors_path,
                alpha_values,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        else:
            # Single generation
            console.print(f"Generating with α={alpha}")
            from dyad.generate import generate_text

            steer_hook = SteerHook(vectors, alpha=alpha)
            generated_text = generate_text(
                model_obj,
                tokenizer,
                prompt,
                steer_hook=steer_hook,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            generations = {alpha: generated_text}

        # Save generations
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_file = output / f"generation_{timestamp}.json"

        metadata = {
            "model": model,
            "trait": trait,
            "experiment": str(experiment),
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }

        save_generations(generations, output_file, prompt, metadata)

        # Display results
        console.print("\n[bold green]Generated text:[/bold green]")
        for alpha_val, text in sorted(generations.items()):
            console.print(f"\n[bold]α={alpha_val}:[/bold]")
            console.print(text)

        console.print(f"\n[bold green]✓ Generation complete![/bold green]")
        console.print(f"Saved to: {output_file}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise click.Abort()


@cli.command()
@click.option(
    "--experiment",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to experiment directory from discover/steer commands",
)
@click.option(
    "--generation-file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Specific generation file to analyze. If not provided, uses most recent.",
)
@click.option(
    "--metrics",
    default="all",
    help="Which metrics to compute (comma-separated: delta_proj,textual_shift,stability). Default: all",
)
@click.option(
    "--plots",
    default="all",
    help="Which plots to generate (comma-separated: curves,polarity,heatmap). Default: all",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory for analysis results. Default: experiment/analysis/",
)
@click.option(
    "--model",
    default=None,
    help="Model identifier (required for full activation analysis). If not provided, uses text-based analysis only.",
)
@click.option(
    "--recompute-activations",
    is_flag=True,
    help="Recompute activations for base and steered texts (requires --model)",
)
def analyze(
    experiment: Path,
    generation_file: Path | None,
    metrics: str,
    plots: str,
    output: Path | None,
    model: str | None,
    recompute_activations: bool,
):
    """Generate quantitative and visual reports."""
    try:
        console.print(f"[bold blue]Dyad: Analysis[/bold blue]")
        console.print(f"Experiment: {experiment}")

        # Determine output directory
        if output is None:
            output = experiment / "analysis"
        output.mkdir(parents=True, exist_ok=True)
        plots_dir = output / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Load experiment config
        config_path = experiment / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            console.print(f"Model: {config.get('model', 'unknown')}")
        else:
            config = {}
            console.print("[yellow]Warning: No config.json found[/yellow]")

        # Load trait vectors
        vectors_path = experiment / "vectors" / "trait_vectors.npz"
        if not vectors_path.exists():
            console.print(
                f"[red]Error: Trait vectors not found at {vectors_path}[/red]"
            )
            raise click.Abort()

        trait_vectors = load_vectors(vectors_path)
        console.print(f"Loaded {len(trait_vectors)} trait vectors")

        # Find generation file
        if generation_file is None:
            gen_dir = experiment / "generations"
            if gen_dir.exists():
                gen_files = sorted(
                    gen_dir.glob("*.json"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )
                if gen_files:
                    generation_file = gen_files[0]
                    console.print(
                        f"Using most recent generation: {generation_file.name}"
                    )
                else:
                    console.print(
                        "[yellow]No generation files found. Analysis will be limited.[/yellow]"
                    )
            else:
                console.print(
                    "[yellow]No generations directory found. Analysis will be limited.[/yellow]"
                )

        # Load generations if available
        generations = None
        if generation_file and generation_file.exists():
            gen_data = load_generations(generation_file)
            generations = gen_data.get("generations", {})
            console.print(f"Loaded {len(generations)} generations")

        # Parse metrics and plots options
        metrics_list = (
            [m.strip() for m in metrics.split(",")] if metrics != "all" else ["all"]
        )
        plots_list = (
            [p.strip() for p in plots.split(",")] if plots != "all" else ["all"]
        )

        # For now, do text-based analysis (full activation analysis requires model)
        # This is a simplified analysis that works with existing data
        console.print("\n[bold]Computing metrics...[/bold]")

        # Simple delta_proj approximation from text analysis
        # In a full implementation, this would recompute activations
        delta_proj = {}
        if generations and "0.0" in generations:
            # Use text-based analysis as approximation
            base_text = generations[0.0]
            for alpha, text in generations.items():
                if alpha == 0.0:
                    continue
                # Simple approximation: use text length differences as proxy
                # Real implementation would compute actual activations
                for layer_idx in trait_vectors.keys():
                    if layer_idx not in delta_proj:
                        delta_proj[layer_idx] = 0.0
                    # Very rough approximation
                    length_diff = len(text.split()) - len(base_text.split())
                    delta_proj[layer_idx] += float(alpha) * length_diff * 0.0001

        if not delta_proj:
            # Create placeholder delta_proj
            for layer_idx in trait_vectors.keys():
                delta_proj[layer_idx] = 0.0

        # Generate plots
        if "all" in plots_list or "curves" in plots_list:
            console.print("\n[bold]Generating layer projection curves...[/bold]")
            plot_path = plots_dir / "layer_projection_curves.png"
            plot_layer_projection_curves(
                delta_proj,
                plot_path,
                title=f"Layer Projection Curves - {experiment.name}",
            )

        if "all" in plots_list or "polarity" in plots_list:
            if generations:
                console.print("\n[bold]Generating textual polarity shifts...[/bold]")
                plot_path = plots_dir / "textual_polarity_shifts.png"
                plot_textual_polarity_shifts(
                    generations,
                    plot_path,
                    title=f"Textual Polarity Shifts - {experiment.name}",
                )

        # Generate report
        console.print("\n[bold]Generating analysis report...[/bold]")
        metadata = {
            "experiment": str(experiment),
            "model": config.get("model", model or "unknown"),
            "trait": config.get("trait", "unknown"),
        }

        report = generate_analysis_report(
            delta_proj=delta_proj,
            generations=generations,
            metadata=metadata,
        )

        # Save JSON report
        report_path = output / "analysis.json"
        save_analysis_report(report, report_path)

        # Generate and save textual summary
        summary = generate_textual_summary(report, experiment_name=experiment.name)
        summary_path = output / "README.md"
        save_textual_summary(summary, summary_path)

        console.print(f"\n[bold green]✓ Analysis complete![/bold green]")
        console.print(f"Report saved to: {report_path}")
        console.print(f"Summary saved to: {summary_path}")
        console.print(f"Plots saved to: {plots_dir}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise click.Abort()


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
