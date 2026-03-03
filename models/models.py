import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Backbone(nn.Module):
    """
    Thin wrapper around torchvision's ResNet‑34 to expose C2–C5 feature maps.

    The first stage (C2) has a stride of 4 for a 640×640 input, producing a
    160×160 feature map. Higher stages downsample further by factors of 2.
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)
        # Stem layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        # Residual stages (C2 through C5)
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning multi‑scale feature maps."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """
    Feature Pyramid Network (1×1 lateral connections + top‑down upsampling).

    Only the highest resolution (P2) is used downstream by the heads, but all
    pyramid levels are returned for potential future use.
    """
    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        # Lateral 1×1 projections from C2–C5 into a fixed number of channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        # 3×3 convolutions applied after merging top‑down features
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c2, c3, c4, c5 = features
        # top‑down pathway
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        # output convolutions
        p5 = self.output_convs[3](p5)
        p4 = self.output_convs[2](p4)
        p3 = self.output_convs[1](p3)
        p2 = self.output_convs[0](p2)
        return p2, p3, p4, p5


class HeatmapHead(nn.Module):
    """
    A simple convolutional head that predicts a single‑channel heatmap from a
    feature map. Consists of a stack of 3×3 convs followed by a 1×1 conv.
    """
    def __init__(self, in_channels: int, num_convs: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.out(x)
        return x


class HybridModel(nn.Module):
    """
    Hybrid localisation + density model.

    Produces two localisation heatmaps (pos/neg), two density heatmaps and
    corresponding scalar counts by summing the softplus of the density maps.
    """
    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained=True)
        self.fpn = FPN(in_channels=[64, 128, 256, 512], out_channels=256)
        # Heads: separate localisation/density for positive and negative classes
        self.loc_pos_head = HeatmapHead(in_channels=256)
        self.loc_neg_head = HeatmapHead(in_channels=256)
        self.count_pos_head = HeatmapHead(in_channels=256)
        self.count_neg_head = HeatmapHead(in_channels=256)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Backbone + FPN
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, _ = self.fpn([c2, c3, c4, c5])  # use highest resolution
        # Predict localisation heatmaps
        loc_pos = self.loc_pos_head(p2)
        loc_neg = self.loc_neg_head(p2)
        # Predict density heatmaps
        den_pos = self.count_pos_head(p2)
        den_neg = self.count_neg_head(p2)
        # Stack channels
        loc_hm = torch.cat([loc_pos, loc_neg], dim=1)
        den_hm = torch.cat([den_pos, den_neg], dim=1)
        # Scalar counts from density maps
        count_pos = F.softplus(den_pos).sum(dim=(2, 3))
        count_neg = F.softplus(den_neg).sum(dim=(2, 3))
        count = torch.cat([count_pos, count_neg], dim=1)
        return loc_hm, den_hm, count