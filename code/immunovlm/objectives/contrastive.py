import torch
from torch import nn
from torch.nn import functional as F


def cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).transpose(0, 1)


def pairwise_spatial_distance(
    coordinates: torch.Tensor, section_codes: torch.Tensor | None = None
) -> torch.Tensor:
    distance = torch.cdist(coordinates.float(), coordinates.float())
    if section_codes is not None:
        different = section_codes[:, None] != section_codes[None, :]
        distance = distance.masked_fill(different, torch.nan)
    return distance


def normalized_distance(distance: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(distance)
    safe = torch.where(finite, distance, torch.zeros_like(distance))
    maximum = safe.amax(dim=1, keepdim=True).clamp_min(torch.finfo(safe.dtype).eps)
    normalized = safe / maximum
    return torch.where(finite, normalized, torch.ones_like(normalized))


class SpatialAwareInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, distance_scale: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.distance_scale = distance_scale

    def forward(
        self,
        vision: torch.Tensor,
        spatial: torch.Tensor,
        coordinates: torch.Tensor,
        section_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        similarity = cosine_matrix(vision, spatial)
        distance = pairwise_spatial_distance(coordinates, section_codes)
        scaled = normalized_distance(distance)
        temperatures = self.temperature * (1.0 + self.distance_scale * scaled)
        logits = similarity / temperatures
        positive = similarity.diagonal() / self.temperature
        diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        negatives = logits.masked_fill(diagonal, -torch.inf)
        denominator = torch.logsumexp(torch.cat([positive[:, None], negatives], dim=1), dim=1)
        image_to_gene = denominator - positive
        reverse_logits = similarity.transpose(0, 1) / temperatures.transpose(0, 1)
        reverse_positive = similarity.diagonal() / self.temperature
        reverse_negative = reverse_logits.masked_fill(diagonal, -torch.inf)
        reverse_denominator = torch.logsumexp(
            torch.cat([reverse_positive[:, None], reverse_negative], dim=1), dim=1
        )
        gene_to_image = reverse_denominator - reverse_positive
        return 0.5 * (image_to_gene.mean() + gene_to_image.mean())


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        logits = cosine_matrix(left, right) / self.temperature
        target = torch.arange(logits.shape[0], device=logits.device)
        forward = F.cross_entropy(logits, target)
        backward = F.cross_entropy(logits.transpose(0, 1), target)
        return 0.5 * (forward + backward)
