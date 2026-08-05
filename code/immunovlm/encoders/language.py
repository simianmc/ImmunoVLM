import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        embedding_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        width = int(self.backbone.config.hidden_size)
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embedding_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.projection(pooled), dim=-1)
