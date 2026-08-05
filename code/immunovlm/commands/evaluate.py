import argparse
import json
from pathlib import Path

import numpy as np

from immunovlm.assessment.bootstrap import bootstrap_interval
from immunovlm.assessment.calibration import calibration_summary, multiclass_brier_score
from immunovlm.assessment.classification import summarize_classification


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="immunovlm-evaluate")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--resamples", type=int, default=2000)
    return value


def run(arguments: argparse.Namespace) -> None:
    archive = np.load(arguments.predictions)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    summary = summarize_classification(labels, probabilities)
    calibration = calibration_summary(labels, probabilities)
    interval = bootstrap_interval(
        labels,
        probabilities,
        lambda target, probability: summarize_classification(target, probability).auroc_macro,
        resamples=arguments.resamples,
    )
    result = {
        "auroc_macro": summary.auroc_macro,
        "auroc_lower": interval.lower,
        "auroc_upper": interval.upper,
        "f1_macro": summary.f1_macro,
        "accuracy": summary.accuracy,
        "specificity_macro": summary.specificity_macro,
        "cohen_kappa": summary.cohen_kappa,
        "expected_calibration_error": calibration.expected_error,
        "maximum_calibration_error": calibration.maximum_error,
        "brier_score": multiclass_brier_score(labels, probabilities),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(arguments.output)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
