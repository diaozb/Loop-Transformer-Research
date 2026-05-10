from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(
    matrix: np.ndarray,
    x_labels: Iterable[int],
    y_labels: Iterable[int],
    output_path: str | Path,
    title: str,
    xlabel: str = "Loop count",
    ylabel: str = "Length",
    colorbar_label: str = "value",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(image, label=colorbar_label)
    x_labels = list(x_labels)
    y_labels = list(y_labels)
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()

