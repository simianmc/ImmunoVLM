from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    half_width: float


def stratified_bootstrap_indices(
    labels: IntArray, resamples: int, generator: np.random.Generator
) -> list[IntArray]:
    classes = np.unique(labels)
    by_class = [np.flatnonzero(labels == value) for value in classes]
    results: list[IntArray] = []
    for _ in range(resamples):
        pieces = [
            generator.choice(indices, size=len(indices), replace=True) for indices in by_class
        ]
        combined = np.concatenate(pieces).astype(np.int64)
        generator.shuffle(combined)
        results.append(combined)
    return results


def bootstrap_interval(
    labels: IntArray,
    values: FloatArray,
    statistic: Callable[[IntArray, FloatArray], float],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    generator = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            statistic(labels[index], values[index])
            for index in stratified_bootstrap_indices(labels, resamples, generator)
        ],
        dtype=np.float64,
    )
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    estimate = statistic(labels, values)
    return Interval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        half_width=float((upper - lower) / 2.0),
    )


def paired_bootstrap_difference(
    labels: IntArray,
    first: FloatArray,
    second: FloatArray,
    statistic: Callable[[IntArray, FloatArray], float],
    resamples: int = 2000,
    seed: int = 42,
) -> Interval:
    generator = np.random.default_rng(seed)
    differences: list[float] = []
    for index in stratified_bootstrap_indices(labels, resamples, generator):
        differences.append(
            statistic(labels[index], first[index]) - statistic(labels[index], second[index])
        )
    values = np.asarray(differences)
    lower, upper = np.quantile(values, [0.025, 0.975])
    estimate = statistic(labels, first) - statistic(labels, second)
    return Interval(float(estimate), float(lower), float(upper), float((upper - lower) / 2.0))
