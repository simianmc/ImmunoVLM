from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DeLongResult:
    auc_first: float
    auc_second: float
    difference: float
    standard_error: float
    z_score: float
    p_value: float
    lower: float
    upper: float


def midrank(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(values.size, dtype=np.float64)
    left = 0
    while left < values.size:
        right = left
        while right < values.size and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[left:right] = 0.5 * (left + right - 1)
        left = right
    result = np.empty(values.size, dtype=np.float64)
    result[order] = ranks + 1.0
    return result


def covariance_matrix(values: FloatArray) -> FloatArray:
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] <= 1:
        return np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    covariance = np.cov(values, bias=False)
    if np.ndim(covariance) == 0:
        return np.asarray([[covariance]], dtype=np.float64)
    return np.asarray(covariance, dtype=np.float64)


def fast_delong(predictions: FloatArray, positive_count: int) -> tuple[FloatArray, FloatArray]:
    classifier_count, example_count = predictions.shape
    negative_count = example_count - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    positive_ranks = np.empty((classifier_count, positive_count), dtype=np.float64)
    negative_ranks = np.empty((classifier_count, negative_count), dtype=np.float64)
    total_ranks = np.empty((classifier_count, example_count), dtype=np.float64)
    for classifier in range(classifier_count):
        positive_ranks[classifier] = midrank(positive[classifier])
        negative_ranks[classifier] = midrank(negative[classifier])
        total_ranks[classifier] = midrank(predictions[classifier])
    aucs = total_ranks[:, :positive_count].sum(axis=1) / (positive_count * negative_count) - (
        positive_count + 1.0
    ) / (2.0 * negative_count)
    positive_components = (total_ranks[:, :positive_count] - positive_ranks) / negative_count
    negative_components = 1.0 - (total_ranks[:, positive_count:] - negative_ranks) / positive_count
    covariance = covariance_matrix(positive_components) / positive_count
    covariance += covariance_matrix(negative_components) / negative_count
    return aucs, covariance


def delong_test(labels: IntArray, first: FloatArray, second: FloatArray) -> DeLongResult:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("DeLong comparison requires binary labels")
    order = np.argsort(-labels)
    positive_count = int(labels.sum())
    predictions = np.stack([first[order], second[order]], axis=0)
    aucs, covariance = fast_delong(predictions, positive_count)
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    standard_error = float(np.sqrt(max(variance, np.finfo(np.float64).eps)))
    difference = float(aucs[0] - aucs[1])
    z_score = difference / standard_error
    p_value = float(2.0 * norm.sf(abs(z_score)))
    lower = difference - 1.959963984540054 * standard_error
    upper = difference + 1.959963984540054 * standard_error
    return DeLongResult(
        auc_first=float(aucs[0]),
        auc_second=float(aucs[1]),
        difference=difference,
        standard_error=standard_error,
        z_score=float(z_score),
        p_value=p_value,
        lower=float(lower),
        upper=float(upper),
    )


def one_vs_rest_delong(
    labels: IntArray, first: FloatArray, second: FloatArray
) -> tuple[DeLongResult, ...]:
    if first.shape != second.shape:
        raise ValueError("probability matrices must have the same shape")
    output: list[DeLongResult] = []
    for class_index in range(first.shape[1]):
        binary = (labels == class_index).astype(np.int64)
        output.append(delong_test(binary, first[:, class_index], second[:, class_index]))
    return tuple(output)


def holm_bonferroni(p_values: FloatArray) -> FloatArray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = p_values.size
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted
