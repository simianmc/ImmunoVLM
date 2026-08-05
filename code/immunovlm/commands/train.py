import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from immunovlm.common.configuration import apply_overrides, load_yaml, materialize_study
from immunovlm.common.randomness import set_seed
from immunovlm.corpora.dataset import TissueSpotDataset, collate_tissue_batch
from immunovlm.corpora.manifest import read_manifest, validate_records
from immunovlm.corpora.splits import patient_level_folds, select_records, verify_disjoint_patients
from immunovlm.encoders.system import ImmunoVLM
from immunovlm.objectives.composite import CompositeObjective
from immunovlm.optimization.distributed import (
    finalize_distributed,
    initialize_distributed,
    wrap_distributed,
)
from immunovlm.optimization.engine import Trainer

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="immunovlm-train")
    value.add_argument("--study", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("override", nargs="*")
    return value


def build_loader(
    dataset: TissueSpotDataset,
    batch_size: int,
    workers: int,
    world_size: int,
    rank: int,
    training: bool,
) -> DataLoader[object]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        drop_last=training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=training,
        collate_fn=collate_tissue_batch,
    )


def run(arguments: argparse.Namespace) -> None:
    configuration = materialize_study(
        apply_overrides(load_yaml(arguments.study), arguments.override), arguments.output
    )
    context = initialize_distributed()
    set_seed(configuration.runtime.seed + context.rank, configuration.runtime.deterministic)
    records = read_manifest(arguments.manifest, arguments.data_root)
    errors = validate_records(records)
    if errors:
        raise ValueError("\n".join(errors[:20]))
    folds = patient_level_folds(records, seed=configuration.runtime.seed)
    selected = folds[configuration.fold]
    verify_disjoint_patients(records, selected)
    tokenizer = AutoTokenizer.from_pretrained(configuration.encoder.language_name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("tokenizer does not expose the required interface")
    training_data = TissueSpotDataset(select_records(records, selected.train), tokenizer, True)
    validation_data = TissueSpotDataset(
        select_records(records, selected.validation), tokenizer, False
    )
    training_loader = build_loader(
        training_data,
        configuration.optimizer.per_device_batch,
        configuration.runtime.workers,
        context.world_size,
        context.rank,
        True,
    )
    validation_loader = build_loader(
        validation_data,
        configuration.optimizer.per_device_batch,
        configuration.runtime.workers,
        context.world_size,
        context.rank,
        False,
    )
    model = ImmunoVLM(configuration.encoder)
    model.freeze_vision()
    wrapped = wrap_distributed(model, context)
    trainer = Trainer(
        wrapped,
        CompositeObjective(configuration.objective),
        configuration.optimizer,
        configuration.runtime,
        context,
        len(training_loader),
    )
    trainer.fit(training_loader, validation_loader)
    finalize_distributed()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run(parser().parse_args())


if __name__ == "__main__":
    main()
