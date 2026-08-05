import torch
from torch import nn

from immunovlm.common.types import EmbeddingSet, EncoderSettings, TissueBatch
from immunovlm.encoders.language import LanguageEncoder
from immunovlm.encoders.spatial import NeighborhoodPool, SpatialTranscriptomeEncoder
from immunovlm.encoders.vision import VisionEncoder


class ImmunoVLM(nn.Module):
    def __init__(self, settings: EncoderSettings, pretrained_vision: bool = True) -> None:
        super().__init__()
        self.vision = VisionEncoder(settings.embedding_dim, pretrained_vision)
        self.spatial = SpatialTranscriptomeEncoder(
            gene_count=settings.gene_count,
            width=settings.spatial_width,
            heads=settings.spatial_heads,
            layers=settings.spatial_layers,
            feedforward=settings.spatial_feedforward,
            embedding_dim=settings.embedding_dim,
            dropout=settings.dropout,
        )
        self.language = LanguageEncoder(settings.language_name, settings.embedding_dim)
        self.neighborhood = NeighborhoodPool(6)

    def forward(self, batch: TissueBatch) -> EmbeddingSet:
        vision = self.vision(batch.images)
        spatial = self.spatial(batch.genes, batch.coordinates)
        language = self.language(batch.input_ids, batch.attention_mask)
        section_codes = self._section_codes(batch.section_ids, spatial.device)
        graph = self.neighborhood(spatial, batch.coordinates, section_codes)
        return EmbeddingSet(vision=vision, spatial=spatial, language=language, graph=graph)

    @staticmethod
    def _section_codes(section_ids: list[str], device: torch.device) -> torch.Tensor:
        mapping: dict[str, int] = {}
        values: list[int] = []
        for section_id in section_ids:
            if section_id not in mapping:
                mapping[section_id] = len(mapping)
            values.append(mapping[section_id])
        return torch.tensor(values, device=device, dtype=torch.long)

    def freeze_vision(self) -> None:
        self.vision.freeze_backbone()

    def unfreeze_vision(self) -> None:
        self.vision.unfreeze_backbone()
