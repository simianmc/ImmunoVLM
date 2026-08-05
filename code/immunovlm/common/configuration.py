from dataclasses import dataclass
from pathlib import Path
from typing import Any

from immunovlm.common.types import (
    EncoderSettings,
    ObjectiveSettings,
    OptimizerSettings,
    RuntimeSettings,
)
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class StudyConfiguration:
    name: str
    fold: int
    encoder: EncoderSettings
    objective: ObjectiveSettings
    optimizer: OptimizerSettings
    runtime: RuntimeSettings


def load_yaml(path: Path) -> DictConfig:
    current = OmegaConf.load(path)
    inherited = current.get("inherits")
    if inherited:
        parent = load_yaml(path.parent / str(inherited))
        current = OmegaConf.merge(parent, current)
    return current


def apply_overrides(config: DictConfig, overrides: list[str]) -> DictConfig:
    if not overrides:
        return config
    return OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))


def materialize_study(config: DictConfig, output_dir: Path) -> StudyConfiguration:
    encoder_values = OmegaConf.to_container(config.encoder, resolve=True)
    objective_values = OmegaConf.to_container(config.objective, resolve=True)
    optimizer_values = OmegaConf.to_container(config.optimizer, resolve=True)
    runtime_values = OmegaConf.to_container(config.runtime, resolve=True)
    if not isinstance(encoder_values, dict):
        raise TypeError("encoder configuration must be a mapping")
    if not isinstance(objective_values, dict):
        raise TypeError("objective configuration must be a mapping")
    if not isinstance(optimizer_values, dict):
        raise TypeError("optimizer configuration must be a mapping")
    if not isinstance(runtime_values, dict):
        raise TypeError("runtime configuration must be a mapping")
    runtime_keys = {"world_size", "workers", "checkpoint_interval", "deterministic"}
    selected_runtime = {key: value for key, value in runtime_values.items() if key in runtime_keys}
    selected_runtime["output_dir"] = output_dir
    selected_runtime["seed"] = int(config.seed)
    return StudyConfiguration(
        name=str(config.study),
        fold=int(config.fold),
        encoder=EncoderSettings(**_typed_dict(encoder_values)),
        objective=ObjectiveSettings(**_typed_dict(objective_values)),
        optimizer=OptimizerSettings(**_typed_dict(optimizer_values)),
        runtime=RuntimeSettings(**selected_runtime),
    )


def _typed_dict(values: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in values.items()}
