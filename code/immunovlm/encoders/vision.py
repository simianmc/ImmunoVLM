import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ViT_L_16_Weights, vit_l_16


class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 1024, pretrained: bool = True) -> None:
        super().__init__()
        weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vit_l_16(weights=weights)
        width = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, embedding_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(images)), dim=-1)

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        self.backbone.requires_grad_(True)
