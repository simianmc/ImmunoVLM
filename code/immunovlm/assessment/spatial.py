import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def inverse_distance_weights(coordinates: FloatArray) -> FloatArray:
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(difference**2, axis=-1))
    weights = np.divide(1.0, distance, out=np.zeros_like(distance), where=distance > 0)
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)


def morans_i(values: FloatArray, coordinates: FloatArray) -> float:
    weights = inverse_distance_weights(coordinates)
    centered = values - values.mean(axis=0, keepdims=True)
    numerator = np.sum(weights[..., None] * centered[:, None, :] * centered[None, :, :])
    denominator = np.sum(centered**2)
    scale = values.shape[0] / np.sum(weights)
    return float(scale * numerator / max(denominator, np.finfo(np.float64).eps))


def spatial_embedding_correlation(embeddings: FloatArray, coordinates: FloatArray) -> float:
    spatial_difference = coordinates[:, None, :] - coordinates[None, :, :]
    spatial_distance = np.sqrt(np.sum(spatial_difference**2, axis=-1))
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    upper = np.triu_indices(embeddings.shape[0], k=1)
    spatial_rank = _rank(spatial_distance[upper])
    embedding_rank = _rank(similarity[upper])
    return float(np.corrcoef(spatial_rank, -embedding_rank)[0, 1])


def _rank(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks
