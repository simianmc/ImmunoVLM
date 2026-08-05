from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatMatrix = npt.NDArray[np.float32]


@dataclass(frozen=True)
class GeneSelection:
    names: tuple[str, ...]
    indices: npt.NDArray[np.int64]
    means: FloatMatrix
    standard_deviations: FloatMatrix


def library_size_normalize(matrix: FloatMatrix, target_sum: float = 10000.0) -> FloatMatrix:
    totals = matrix.sum(axis=1, keepdims=True)
    safe = np.maximum(totals, np.finfo(np.float32).eps)
    return (matrix / safe * target_sum).astype(np.float32)


def log_transform(matrix: FloatMatrix) -> FloatMatrix:
    return np.log1p(matrix).astype(np.float32)


def standardized_variance(matrix: FloatMatrix, bins: int = 20) -> FloatMatrix:
    means = matrix.mean(axis=0)
    variances = matrix.var(axis=0, ddof=1)
    quantiles = np.quantile(means, np.linspace(0.0, 1.0, bins + 1))
    scores = np.zeros_like(variances, dtype=np.float32)
    for lower, upper in pairwise(quantiles):
        mask = (means >= lower) & (means <= upper)
        selected = variances[mask]
        if selected.size == 0:
            continue
        center = selected.mean()
        scale = selected.std()
        scores[mask] = (selected - center) / max(float(scale), np.finfo(np.float32).eps)
    return scores


def select_highly_variable_genes(
    matrix: FloatMatrix, names: list[str], count: int = 500
) -> GeneSelection:
    if matrix.shape[1] != len(names):
        raise ValueError("gene names do not match expression columns")
    normalized = log_transform(library_size_normalize(matrix))
    score = standardized_variance(normalized)
    indices = np.argsort(score)[-count:][::-1].astype(np.int64)
    selected = normalized[:, indices]
    means = selected.mean(axis=0, keepdims=True).astype(np.float32)
    deviations = selected.std(axis=0, keepdims=True).astype(np.float32)
    deviations = np.maximum(deviations, np.finfo(np.float32).eps)
    return GeneSelection(
        names=tuple(names[index] for index in indices),
        indices=indices,
        means=means,
        standard_deviations=deviations,
    )


def apply_gene_selection(matrix: FloatMatrix, selection: GeneSelection) -> FloatMatrix:
    normalized = log_transform(library_size_normalize(matrix))
    selected = normalized[:, selection.indices]
    return ((selected - selection.means) / selection.standard_deviations).astype(np.float32)


def load_expression(path: Path) -> FloatMatrix:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix == ".npz":
        archive = np.load(path)
        key = "expression" if "expression" in archive else archive.files[0]
        return np.asarray(archive[key], dtype=np.float32)
    raise ValueError(f"unsupported expression format {path.suffix}")


def save_gene_selection(selection: GeneSelection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            names=np.asarray(selection.names),
            indices=selection.indices,
            means=selection.means,
            standard_deviations=selection.standard_deviations,
        )
    temporary.replace(path)


def load_gene_selection(path: Path) -> GeneSelection:
    archive = np.load(path)
    return GeneSelection(
        names=tuple(str(value) for value in archive["names"]),
        indices=np.asarray(archive["indices"], dtype=np.int64),
        means=np.asarray(archive["means"], dtype=np.float32),
        standard_deviations=np.asarray(archive["standard_deviations"], dtype=np.float32),
    )
