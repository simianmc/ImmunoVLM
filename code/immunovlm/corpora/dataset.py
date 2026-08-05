from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from immunovlm.common.types import TissueBatch
from immunovlm.corpora.images import PatchTransform
from immunovlm.corpora.records import SUBTYPE_DESCRIPTIONS, SpotRecord, class_index_map


@dataclass(frozen=True)
class SpotItem:
    image: torch.Tensor
    genes: torch.Tensor
    coordinates: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: int
    section_id: str
    sample_id: str


class ExpressionStore:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(paths)
        self.cache: dict[Path, np.ndarray] = {}

    def get(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        if path not in self.cache:
            value = np.load(path, mmap_mode="r")
            self.cache[path] = np.asarray(value, dtype=np.float32)
        array = self.cache[path]
        if array.ndim != 1:
            raise ValueError(f"spot expression must be one dimensional: {path}")
        return torch.from_numpy(np.array(array, copy=True))


class TissueSpotDataset(Dataset[SpotItem]):
    def __init__(
        self,
        records: Sequence[SpotRecord],
        tokenizer: PreTrainedTokenizerBase,
        training: bool,
        maximum_text_length: int = 96,
    ) -> None:
        eligible = [record for record in records if record.tissue_fraction >= 0.5]
        if not eligible:
            raise ValueError("no eligible tissue spots")
        self.records = tuple(eligible)
        self.tokenizer = tokenizer
        self.transform = PatchTransform(training=training)
        self.maximum_text_length = maximum_text_length
        self.classes = class_index_map([record.subtype for record in self.records])
        self.expression = ExpressionStore([record.expression_path for record in self.records])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SpotItem:
        record = self.records[index]
        with Image.open(record.image_path) as opened:
            image = self.transform(opened.convert("RGB"))
        genes = self.expression.get(index)
        coordinates = torch.tensor([record.coordinate_x, record.coordinate_y], dtype=torch.float32)
        description = SUBTYPE_DESCRIPTIONS[record.subtype]
        tokens = self.tokenizer(
            description,
            max_length=self.maximum_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return SpotItem(
            image=image,
            genes=genes,
            coordinates=coordinates,
            input_ids=tokens["input_ids"].squeeze(0),
            attention_mask=tokens["attention_mask"].squeeze(0),
            label=self.classes[record.subtype],
            section_id=record.section_id,
            sample_id=record.sample_id,
        )


def collate_tissue_batch(items: list[SpotItem]) -> TissueBatch:
    return TissueBatch(
        images=torch.stack([item.image for item in items]),
        genes=torch.stack([item.genes for item in items]),
        coordinates=torch.stack([item.coordinates for item in items]),
        input_ids=torch.stack([item.input_ids for item in items]),
        attention_mask=torch.stack([item.attention_mask for item in items]),
        labels=torch.tensor([item.label for item in items], dtype=torch.long),
        section_ids=[item.section_id for item in items],
        sample_ids=[item.sample_id for item in items],
    )
