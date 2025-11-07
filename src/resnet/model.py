import torch.nn as nn
import torchvision.models as models

def build_resnet50(num_classes: int, freeze_backbone: bool, weights="IMAGENET1K_V2"):
    backbone = models.resnet50(weights=getattr(models.ResNet50_Weights, weights))
    in_feats = backbone.fc.in_features
    backbone.fc = nn.Linear(in_feats, num_classes)

    if freeze_backbone:
        # freeze all except final fc
        for name, p in backbone.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return backbone
