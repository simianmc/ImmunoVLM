import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialPositionEncoding(nn.Module):
    def __init__(self, width: int = 128, omega: float = 10000.0) -> None:
        super().__init__()
        if width % 4 != 0:
            raise ValueError("width must be divisible by four")
        frequencies = omega ** (torch.arange(width // 4, dtype=torch.float32) * 4.0 / width)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.width = width

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        scaled_x = coordinates[..., 0:1] / self.frequencies
        scaled_y = coordinates[..., 1:2] / self.frequencies
        return torch.cat([scaled_x.sin(), scaled_x.cos(), scaled_y.sin(), scaled_y.cos()], dim=-1)


class DifferentialGeneGate(nn.Module):
    def __init__(self, gene_count: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(gene_count))
        self.context = nn.Sequential(
            nn.Linear(gene_count, gene_count),
            nn.GELU(),
            nn.Linear(gene_count, gene_count),
            nn.Sigmoid(),
        )

    def forward(self, genes: torch.Tensor) -> torch.Tensor:
        fixed = torch.softmax(self.logits, dim=0) * self.logits.numel()
        adaptive = self.context(genes)
        return genes * fixed * adaptive

    def importance(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)


class GeneTokenization(nn.Module):
    def __init__(self, gene_count: int, width: int) -> None:
        super().__init__()
        self.values = nn.Linear(1, width)
        self.identities = nn.Embedding(gene_count, width)
        self.register_buffer("indices", torch.arange(gene_count), persistent=False)
        self.normalization = nn.LayerNorm(width)

    def forward(self, genes: torch.Tensor) -> torch.Tensor:
        tokens = self.values(genes.unsqueeze(-1))
        tokens = tokens + self.identities(self.indices).unsqueeze(0)
        return self.normalization(tokens)


class SpatialTranscriptomeEncoder(nn.Module):
    def __init__(
        self,
        gene_count: int = 500,
        width: int = 128,
        heads: int = 8,
        layers: int = 4,
        feedforward: int = 512,
        embedding_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate = DifferentialGeneGate(gene_count)
        self.tokenization = GeneTokenization(gene_count, width)
        self.position = SpatialPositionEncoding(width)
        self.position_projection = nn.Linear(width, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.pool_query = nn.Parameter(torch.empty(width))
        self.pool_projection = nn.Linear(width, width)
        self.output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, embedding_dim),
        )
        nn.init.normal_(self.pool_query, std=1.0 / math.sqrt(width))

    def forward(self, genes: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        gated = self.gate(genes)
        tokens = self.tokenization(gated)
        position = self.position_projection(self.position(coordinates)).unsqueeze(1)
        encoded = self.transformer(tokens + position)
        scores = self.pool_projection(encoded) @ self.pool_query
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return F.normalize(self.output(pooled), dim=-1)

    def gene_importance(self) -> torch.Tensor:
        return self.gate.importance()


class NeighborhoodPool(nn.Module):
    def __init__(self, neighbors: int = 6) -> None:
        super().__init__()
        self.neighbors = neighbors

    def forward(
        self,
        embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        section_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        distance = torch.cdist(coordinates.float(), coordinates.float())
        diagonal = torch.eye(distance.shape[0], device=distance.device, dtype=torch.bool)
        distance = distance.masked_fill(diagonal, torch.inf)
        if section_codes is not None:
            cross_section = section_codes[:, None] != section_codes[None, :]
            distance = distance.masked_fill(cross_section, torch.inf)
        count = min(self.neighbors, max(1, distance.shape[1] - 1))
        indices = distance.topk(count, largest=False, dim=1).indices
        neighbors = embeddings[indices]
        finite = torch.isfinite(distance.gather(1, indices)).unsqueeze(-1)
        denominator = finite.sum(dim=1).clamp_min(1)
        pooled = (neighbors * finite).sum(dim=1) / denominator
        return F.normalize(pooled, dim=-1)
