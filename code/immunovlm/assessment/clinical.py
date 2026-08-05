from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DecisionPoint:
    threshold: float
    net_benefit: float
    treat_all: float
    treat_none: float


def net_benefit(labels: IntArray, risks: FloatArray, threshold: float) -> float:
    predicted = risks >= threshold
    positive = labels == 1
    true_positive = np.sum(predicted & positive)
    false_positive = np.sum(predicted & ~positive)
    odds = threshold / (1.0 - threshold)
    return float(true_positive / labels.size - false_positive / labels.size * odds)


def decision_curve(
    labels: IntArray,
    risks: FloatArray,
    thresholds: FloatArray | None = None,
) -> tuple[DecisionPoint, ...]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 20)
    prevalence = float((labels == 1).mean())
    output: list[DecisionPoint] = []
    for threshold in thresholds:
        treat_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
        output.append(
            DecisionPoint(
                threshold=float(threshold),
                net_benefit=net_benefit(labels, risks, float(threshold)),
                treat_all=float(treat_all),
                treat_none=0.0,
            )
        )
    return tuple(output)


def category_free_nri(labels: IntArray, reference: FloatArray, candidate: FloatArray) -> float:
    event = labels == 1
    non_event = ~event
    event_up = np.mean(candidate[event] > reference[event])
    event_down = np.mean(candidate[event] < reference[event])
    non_event_down = np.mean(candidate[non_event] < reference[non_event])
    non_event_up = np.mean(candidate[non_event] > reference[non_event])
    return float(event_up - event_down + non_event_down - non_event_up)


def operating_point(labels: IntArray, risks: FloatArray, mode: str) -> tuple[float, float, float]:
    thresholds = np.unique(risks)
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        prediction = risks >= threshold
        positive = labels == 1
        negative = ~positive
        sensitivity = float(np.sum(prediction & positive) / max(1, np.sum(positive)))
        specificity = float(np.sum(~prediction & negative) / max(1, np.sum(negative)))
        candidates.append((float(threshold), sensitivity, specificity))
    if mode == "high_sensitivity":
        return min(candidates, key=lambda item: (abs(item[1] - 0.95), -item[2]))
    if mode == "high_specificity":
        return min(candidates, key=lambda item: (abs(item[2] - 0.95), -item[1]))
    if mode == "youden":
        return max(candidates, key=lambda item: item[1] + item[2] - 1.0)
    raise ValueError(mode)
