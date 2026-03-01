
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)

        # Stem
        self.conv1 = resnet.conv1
        self.bn1   = resnet.bn1
        self.relu  = resnet.relu
        self.maxpool = resnet.maxpool

        # Residual stages
        self.layer1 = resnet.layer1  # C2 (stride 4)
        self.layer2 = resnet.layer2  # C3 (stride 8)
        self.layer3 = resnet.layer3  # C4 (stride 16)
        self.layer4 = resnet.layer4  # C5 (stride 32)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        c2 = self.layer1(x)  # 160×160
        c3 = self.layer2(c2) # 80×80
        c4 = self.layer3(c3) # 40×40
        c5 = self.layer4(c4) # 20×20

        return c2, c3, c4, c5


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features):
        # features = [C2, C3, C4, C5]
        c2, c3, c4, c5 = features

        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        p5 = self.output_convs[3](p5)
        p4 = self.output_convs[2](p4)
        p3 = self.output_convs[1](p3)
        p2 = self.output_convs[0](p2)

        return p2, p3, p4, p5



class HeatmapHead(nn.Module):
    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),    # keeps HW same
                nn.ReLU(inplace=True)
            ]
        self.conv = nn.Sequential(*layers)
        self.out  = nn.Conv2d(in_channels, 1, kernel_size=1)                      # # keeps HW same

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        return x




class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained=True)
        self.fpn = FPN(
            in_channels=[64, 128, 256, 512],  # ResNet-34 C2–C5
            out_channels=256
        )

        self.loc_pos_head = HeatmapHead(in_channels=256)
        self.loc_neg_head = HeatmapHead(in_channels=256)

        self.count_pos_head = HeatmapHead(in_channels=256)
        self.count_neg_head = HeatmapHead(in_channels=256)

    def forward(self, x):
        '''
        Docstring for forward

        param:
        x: tensor of images (batch_size, 3, 640, 640)
        
        returns:
        
        loc_hm: stacked localization heatmaps for positive and negative cells, shape: (B, 2, 160, 160)
        den_hm: stacked density heatmaps for positive and negative cells, shape: (B, 2, 160, 160)
        count: positive and negative cell counts, shape: (B, 2)
        '''
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, _ = self.fpn([c2, c3, c4, c5])   # (B, 256, 160, 160)

        # Localization heatmaps (trained vs H_max targets)
        loc_pos = self.loc_pos_head(p2)  # (B, 1, 160, 160)
        loc_neg = self.loc_neg_head(p2)  # (B, 1, 160, 160)

        # Density heatmaps (trained vs H_sum targets)
        den_pos = self.count_pos_head(p2)  # (B, 1, 160, 160)
        den_neg = self.count_neg_head(p2)  # (B, 1, 160, 160)

        loc_hm = torch.cat([loc_pos, loc_neg], dim=1)  # (B, 2, 160, 160)
        den_hm = torch.cat([den_pos, den_neg], dim=1)  # (B, 2, 160, 160)

        # scalar count from density maps
        count_pos = F.softplus(den_pos).sum(dim=(2,3))
        count_neg = F.softplus(den_neg).sum(dim=(2,3))
        count = torch.cat([count_pos, count_neg], dim=1)   # (B, 2)

        return loc_hm, den_hm, count