import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "best_densenet_model.pth"

class DenseNetHead(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes=10, dropout=0.6):
        super().__init__()
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

try:
    backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
except:
    backbone = models.densenet121(pretrained=True)

feat_dim = backbone.classifier.in_features
model = DenseNetHead(backbone, feat_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

example = torch.randn(1,3,224,224)
traced = torch.jit.trace(model, example)
traced.save("best_densenet_ts.pt")

print("Saved TorchScript model → best_densenet_ts.pt")
