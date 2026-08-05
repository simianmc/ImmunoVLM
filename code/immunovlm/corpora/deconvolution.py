from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import nnls

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DeconvolutionResult:
    proportions: FloatArray
    reconstructed_expression: FloatArray
    residual_norms: FloatArray


def cell_type_reference(
    expression: FloatArray, labels: npt.NDArray[np.str_]
) -> tuple[FloatArray, tuple[str, ...]]:
    classes = tuple(str(value) for value in np.unique(labels))
    profiles = []
    for class_name in classes:
        selected = expression[labels == class_name]
        profiles.append(np.mean(selected, axis=0))
    reference = np.stack(profiles, axis=1)
    scale = np.linalg.norm(reference, axis=0, keepdims=True)
    reference = np.divide(reference, scale, out=np.zeros_like(reference), where=scale > 0)
    return reference, classes


def nonnegative_deconvolution(
    spot_expression: FloatArray, reference: FloatArray
) -> DeconvolutionResult:
    proportions = np.zeros((spot_expression.shape[0], reference.shape[1]), dtype=np.float64)
    residuals = np.zeros(spot_expression.shape[0], dtype=np.float64)
    for index, target in enumerate(spot_expression):
        weights, residual = nnls(reference, target)
        total = weights.sum()
        if total > 0:
            weights /= total
        proportions[index] = weights
        residuals[index] = residual
    reconstructed = proportions @ reference.T
    return DeconvolutionResult(proportions, reconstructed, residuals)


def smooth_proportions(
    proportions: FloatArray,
    coordinates: FloatArray,
    neighbors: int = 6,
    strength: float = 0.25,
) -> FloatArray:
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(difference**2, axis=-1))
    np.fill_diagonal(distance, np.inf)
    indices = np.argpartition(distance, kth=min(neighbors, distance.shape[1] - 1), axis=1)[
        :, :neighbors
    ]
    neighborhood = np.mean(proportions[indices], axis=1)
    combined = (1.0 - strength) * proportions + strength * neighborhood
    totals = combined.sum(axis=1, keepdims=True)
    return np.divide(combined, totals, out=np.zeros_like(combined), where=totals > 0)


def atlas_coordinates(
    proportions: FloatArray,
    reference_coordinates: FloatArray,
    reference_proportions: FloatArray,
    neighbors: int = 6,
) -> FloatArray:
    normalized_query = proportions / np.maximum(
        np.linalg.norm(proportions, axis=1, keepdims=True), 1e-12
    )
    normalized_reference = reference_proportions / np.maximum(
        np.linalg.norm(reference_proportions, axis=1, keepdims=True), 1e-12
    )
    similarity = normalized_query @ normalized_reference.T
    count = min(neighbors, reference_coordinates.shape[0])
    indices = np.argpartition(-similarity, kth=count - 1, axis=1)[:, :count]
    weights = np.take_along_axis(similarity, indices, axis=1)
    weights = np.maximum(weights, 0.0)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    coordinates = np.sum(reference_coordinates[indices] * weights[..., None], axis=1)
    return coordinates


def reconstruction_quality(observed: FloatArray, reconstructed: FloatArray) -> FloatArray:
    centered_observed = observed - observed.mean(axis=1, keepdims=True)
    centered_reconstructed = reconstructed - reconstructed.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_observed * centered_reconstructed, axis=1)
    denominator = np.linalg.norm(centered_observed, axis=1) * np.linalg.norm(
        centered_reconstructed, axis=1
    )
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
