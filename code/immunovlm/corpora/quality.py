from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SpotQuality:
    total_counts: FloatArray
    detected_genes: IntArray
    mitochondrial_fraction: FloatArray
    ribosomal_fraction: FloatArray
    retained: npt.NDArray[np.bool_]


def compute_spot_quality(
    expression: FloatArray,
    gene_names: list[str],
    minimum_genes: int = 200,
    maximum_mitochondrial_fraction: float = 0.25,
) -> SpotQuality:
    totals = expression.sum(axis=1)
    detected = np.sum(expression > 0, axis=1).astype(np.int64)
    mitochondrial = np.asarray([name.upper().startswith("MT-") for name in gene_names])
    ribosomal = np.asarray(
        [name.upper().startswith("RPS") or name.upper().startswith("RPL") for name in gene_names]
    )
    mitochondrial_counts = expression[:, mitochondrial].sum(axis=1)
    ribosomal_counts = expression[:, ribosomal].sum(axis=1)
    safe = np.maximum(totals, np.finfo(np.float64).eps)
    mitochondrial_fraction = mitochondrial_counts / safe
    ribosomal_fraction = ribosomal_counts / safe
    retained = (
        (detected >= minimum_genes)
        & (mitochondrial_fraction <= maximum_mitochondrial_fraction)
        & (totals > 0)
    )
    return SpotQuality(
        total_counts=totals,
        detected_genes=detected,
        mitochondrial_fraction=mitochondrial_fraction,
        ribosomal_fraction=ribosomal_fraction,
        retained=retained,
    )


def median_absolute_deviation(values: FloatArray) -> float:
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def outlier_mask(values: FloatArray, deviations: float = 5.0) -> npt.NDArray[np.bool_]:
    center = np.median(values)
    scale = median_absolute_deviation(values)
    if scale == 0.0:
        return np.zeros(values.shape, dtype=np.bool_)
    return np.abs(values - center) > deviations * 1.4826 * scale


def section_quality_summary(quality: SpotQuality) -> dict[str, float]:
    return {
        "spots": float(quality.total_counts.size),
        "retained_fraction": float(quality.retained.mean()),
        "median_counts": float(np.median(quality.total_counts)),
        "median_detected_genes": float(np.median(quality.detected_genes)),
        "median_mitochondrial_fraction": float(np.median(quality.mitochondrial_fraction)),
        "median_ribosomal_fraction": float(np.median(quality.ribosomal_fraction)),
    }
