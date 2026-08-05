from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class EncoderSettings:
    embedding_dim: int = 1024
    gene_count: int = 500
    spatial_width: int = 128
    spatial_heads: int = 8
    spatial_layers: int = 4
    spatial_feedforward: int = 512
    image_size: int = 224
    dropout: float = 0.1
    vision_name: str = "vit_large_patch16_224"
    language_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


@dataclass(frozen=True)
class ObjectiveSettings:
    base_temperature: float = 0.07
    distance_scale: float = 0.5
    spatial_temperature: float = 0.1
    embedding_temperature: float = 0.1
    topology_weight: float = 0.3
    graph_weight: float = 0.1
    language_weight: float = 0.2
    neighbor_count: int = 6


@dataclass(frozen=True)
class OptimizerSettings:
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 100
    per_device_batch: int = 512
    gradient_accumulation: int = 4
    gradient_clip: float = 1.0
    warmup_steps: int = 0
    precision: Literal["fp16", "fp32"] = "fp16"


@dataclass(frozen=True)
class RuntimeSettings:
    output_dir: Path
    seed: int = 42
    world_size: int = 4
    workers: int = 16
    checkpoint_interval: int = 10
    deterministic: bool = True


@dataclass
class TissueBatch:
    images: torch.Tensor
    genes: torch.Tensor
    coordinates: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    section_ids: list[str]
    sample_ids: list[str]

    def to(self, device: torch.device) -> "TissueBatch":
        return TissueBatch(
            images=self.images.to(device, non_blocking=True),
            genes=self.genes.to(device, non_blocking=True),
            coordinates=self.coordinates.to(device, non_blocking=True),
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            section_ids=self.section_ids,
            sample_ids=self.sample_ids,
        )


@dataclass
class EmbeddingSet:
    vision: torch.Tensor
    spatial: torch.Tensor
    language: torch.Tensor
    graph: torch.Tensor


@dataclass
class LossSet:
    total: torch.Tensor
    spatial: torch.Tensor
    topology: torch.Tensor
    graph: torch.Tensor
    language: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "spatial": float(self.spatial.detach().cpu()),
            "topology": float(self.topology.detach().cpu()),
            "graph": float(self.graph.detach().cpu()),
            "language": float(self.language.detach().cpu()),
        }


@dataclass
class EpochState:
    epoch: int = 0
    global_step: int = 0
    best_validation_loss: float = float("inf")
    histories: dict[str, list[float]] = field(default_factory=dict)
