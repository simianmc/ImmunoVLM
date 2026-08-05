from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HeterogeneitySummary:
    pooled_estimate: float
    cochran_q: float
    degrees_of_freedom: int
    p_value: float
    i_squared: float


def logit(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def inverse_logit(values: FloatArray | float) -> FloatArray | float:
    return 1.0 / (1.0 + np.exp(-values))


def heterogeneity(aurocs: FloatArray, standard_errors: FloatArray) -> HeterogeneitySummary:
    transformed = logit(aurocs)
    derivative = 1.0 / np.clip(aurocs * (1.0 - aurocs), 1e-8, None)
    transformed_variance = (standard_errors * derivative) ** 2
    weights = 1.0 / np.clip(transformed_variance, 1e-12, None)
    pooled_logit = float(np.sum(weights * transformed) / np.sum(weights))
    q = float(np.sum(weights * (transformed - pooled_logit) ** 2))
    degrees = max(0, aurocs.size - 1)
    p_value = float(chi2.sf(q, degrees)) if degrees else 1.0
    i_squared = max(0.0, (q - degrees) / q * 100.0) if q > 0.0 else 0.0
    return HeterogeneitySummary(
        pooled_estimate=float(inverse_logit(pooled_logit)),
        cochran_q=q,
        degrees_of_freedom=degrees,
        p_value=p_value,
        i_squared=i_squared,
    )
