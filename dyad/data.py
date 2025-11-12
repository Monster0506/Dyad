"""Dataset loaders for contrastive trait pairs."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def load_contrastive_pairs(path: Path) -> tuple[list[str], list[str]]:
    """Load contrastive pairs from JSON file.

    Args:
        path: Path to JSON file with contrastive pairs

    Returns:
        Tuple of (positive examples, negative examples) as lists of strings

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON schema is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate schema
    required_keys = {"trait", "positive_label", "negative_label", "pairs"}
    if not all(key in data for key in required_keys):
        raise ValueError(
            f"Invalid schema: missing required keys. Expected: {required_keys}, got: {set(data.keys())}"
        )

    if not isinstance(data["pairs"], list):
        raise ValueError("'pairs' must be a list")

    positive_label = data["positive_label"]
    negative_label = data["negative_label"]

    positive_examples = []
    negative_examples = []

    for pair in data["pairs"]:
        if not isinstance(pair, dict):
            raise ValueError("Each pair must be a dictionary")
        if positive_label not in pair or negative_label not in pair:
            raise ValueError(
                f"Pair missing required labels: {positive_label} or {negative_label}"
            )
        positive_examples.append(pair[positive_label])
        negative_examples.append(pair[negative_label])

    console.print(f"[green]Loaded {len(positive_examples)} contrastive pairs[/green]")
    return positive_examples, negative_examples


def save_dataset(data: dict[str, Any], path: Path) -> None:
    """Save dataset to JSON file.

    Args:
        data: Dataset dictionary with trait structure
        path: Path where to save the JSON file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Saved dataset to {path}[/green]")
