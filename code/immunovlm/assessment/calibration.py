from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float


@dataclass(frozen=True)
class CalibrationSummary:
    expected_error: float
    maximum_error: float
    bins: tuple[CalibrationBin, ...]


def calibration_summary(
    labels: IntArray, probabilities: FloatArray, bin_count: int = 10
) -> CalibrationSummary:
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    output: list[CalibrationBin] = []
    expected = 0.0
    maximum = 0.0
    for index, (lower, upper) in enumerate(pairwise(edges)):
        if index == bin_count - 1:
            selected = (confidences >= lower) & (confidences <= upper)
        else:
            selected = (confidences >= lower) & (confidences < upper)
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else 0.0
        confidence = float(confidences[selected].mean()) if count else 0.0
        difference = abs(accuracy - confidence)
        expected += count / max(1, labels.size) * difference
        maximum = max(maximum, difference)
        output.append(CalibrationBin(float(lower), float(upper), count, accuracy, confidence))
    return CalibrationSummary(float(expected), float(maximum), tuple(output))


def multiclass_brier_score(labels: IntArray, probabilities: FloatArray) -> float:
    target = np.zeros_like(probabilities)
    target[np.arange(labels.size), labels] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))
