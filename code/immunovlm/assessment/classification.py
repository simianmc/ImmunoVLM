from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ClassificationSummary:
    auroc_macro: float
    f1_macro: float
    accuracy: float
    specificity_macro: float
    cohen_kappa: float


def one_hot(labels: IntArray, classes: int) -> FloatArray:
    result = np.zeros((labels.shape[0], classes), dtype=np.float64)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def macro_specificity(labels: IntArray, predictions: IntArray, classes: int) -> float:
    values: list[float] = []
    for class_index in range(classes):
        negative = labels != class_index
        true_negative = np.sum(negative & (predictions != class_index))
        false_positive = np.sum(negative & (predictions == class_index))
        denominator = true_negative + false_positive
        values.append(float(true_negative / denominator) if denominator else 0.0)
    return float(np.mean(values))


def summarize_classification(labels: IntArray, probabilities: FloatArray) -> ClassificationSummary:
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    classes = probabilities.shape[1]
    targets = one_hot(labels, classes)
    auroc = roc_auc_score(targets, probabilities, average="macro", multi_class="ovr")
    return ClassificationSummary(
        auroc_macro=float(auroc),
        f1_macro=float(f1_score(labels, predictions, average="macro")),
        accuracy=float(accuracy_score(labels, predictions)),
        specificity_macro=macro_specificity(labels, predictions, classes),
        cohen_kappa=float(cohen_kappa_score(labels, predictions)),
    )


def confusion_distribution(labels: IntArray, predictions: IntArray, classes: int) -> FloatArray:
    matrix = np.zeros((classes, classes), dtype=np.float64)
    np.add.at(matrix, (labels, predictions), 1.0)
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
