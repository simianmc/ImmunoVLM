import torch
from torch import nn

from immunovlm.common.types import EmbeddingSet, LossSet, ObjectiveSettings, TissueBatch
from immunovlm.objectives.contrastive import SpatialAwareInfoNCE, SymmetricInfoNCE
from immunovlm.objectives.topology import TopologyDivergence


class CompositeObjective(nn.Module):
    def __init__(self, settings: ObjectiveSettings) -> None:
        super().__init__()
        self.settings = settings
        self.spatial_loss = SpatialAwareInfoNCE(settings.base_temperature, settings.distance_scale)
        self.topology_loss = TopologyDivergence(
            settings.spatial_temperature, settings.embedding_temperature
        )
        self.graph_loss = SymmetricInfoNCE(settings.base_temperature)
        self.language_loss = SymmetricInfoNCE(settings.base_temperature)

    def forward(self, embeddings: EmbeddingSet, batch: TissueBatch) -> LossSet:
        section_codes = self._section_codes(batch.section_ids, embeddings.vision.device)
        spatial = self.spatial_loss(
            embeddings.vision, embeddings.spatial, batch.coordinates, section_codes
        )
        topology = self.topology_loss(embeddings.spatial, batch.coordinates, section_codes)
        graph = self.graph_loss(embeddings.spatial, embeddings.graph)
        language = self.language_loss(embeddings.language, embeddings.vision)
        total = (
            spatial
            + self.settings.topology_weight * topology
            + self.settings.graph_weight * graph
            + self.settings.language_weight * language
        )
        return LossSet(total, spatial, topology, graph, language)

    @staticmethod
    def _section_codes(section_ids: list[str], device: torch.device) -> torch.Tensor:
        mapping: dict[str, int] = {}
        result: list[int] = []
        for value in section_ids:
            result.append(mapping.setdefault(value, len(mapping)))
        return torch.tensor(result, dtype=torch.long, device=device)
