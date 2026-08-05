from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from immunovlm.corpora.records import SpotRecord


@dataclass(frozen=True)
class Fold:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def patient_level_folds(
    records: list[SpotRecord], folds: int = 5, seed: int = 42
) -> tuple[Fold, ...]:
    labels = np.asarray([record.subtype for record in records])
    groups = np.asarray([record.patient_id for record in records])
    indices = np.arange(len(records))
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: list[Fold] = []
    for fold_index, (development, test) in enumerate(outer.split(indices, labels, groups)):
        development_labels = labels[development]
        development_groups = groups[development]
        inner = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed + fold_index + 1
        )
        train_local, validation_local = next(
            inner.split(development, development_labels, development_groups)
        )
        results.append(
            Fold(
                train=tuple(int(value) for value in development[train_local]),
                validation=tuple(int(value) for value in development[validation_local]),
                test=tuple(int(value) for value in test),
            )
        )
    return tuple(results)


def select_records(records: list[SpotRecord], indices: tuple[int, ...]) -> list[SpotRecord]:
    return [records[index] for index in indices]


def verify_disjoint_patients(records: list[SpotRecord], fold: Fold) -> None:
    train = {records[index].patient_id for index in fold.train}
    validation = {records[index].patient_id for index in fold.validation}
    test = {records[index].patient_id for index in fold.test}
    if train & validation or train & test or validation & test:
        raise ValueError("patient leakage detected")
