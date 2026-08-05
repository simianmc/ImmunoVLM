import torch
from torch import nn
from torch.nn import functional as F

from immunovlm.objectives.contrastive import cosine_matrix, pairwise_spatial_distance


def median_finite_distance(distance: torch.Tensor) -> torch.Tensor:
    finite = distance[torch.isfinite(distance) & (distance > 0)]
    if finite.numel() == 0:
        return torch.ones((), device=distance.device, dtype=distance.dtype)
    return finite.median().clamp_min(torch.finfo(distance.dtype).eps)


def spatial_similarity(
    coordinates: torch.Tensor, section_codes: torch.Tensor | None = None
) -> torch.Tensor:
    distance = pairwise_spatial_distance(coordinates, section_codes)
    sigma = median_finite_distance(distance)
    similarity = torch.exp(-(distance.square()) / (2.0 * sigma.square()))
    return torch.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)


class TopologyDivergence(nn.Module):
    def __init__(
        self, spatial_temperature: float = 0.1, embedding_temperature: float = 0.1
    ) -> None:
        super().__init__()
        self.spatial_temperature = spatial_temperature
        self.embedding_temperature = embedding_temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        section_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_similarity = spatial_similarity(coordinates, section_codes)
        learned_similarity = cosine_matrix(embeddings, embeddings)
        diagonal = torch.eye(
            learned_similarity.shape[0], dtype=torch.bool, device=learned_similarity.device
        )
        target_logits = (target_similarity / self.spatial_temperature).masked_fill(
            diagonal, -torch.inf
        )
        learned_logits = (learned_similarity / self.embedding_temperature).masked_fill(
            diagonal, -torch.inf
        )
        target = torch.softmax(target_logits, dim=-1)
        learned = torch.log_softmax(learned_logits, dim=-1)
        return F.kl_div(learned, target, reduction="batchmean")


def topology_correlation(embeddings: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    spatial = spatial_similarity(coordinates)
    learned = cosine_matrix(embeddings, embeddings)
    mask = ~torch.eye(spatial.shape[0], dtype=torch.bool, device=spatial.device)
    left = spatial[mask]
    right = learned[mask]
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    return (left @ right) / denominator.clamp_min(torch.finfo(left.dtype).eps)
