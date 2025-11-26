import io
import json
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# CONFIG
MODEL_PATH = "./best_densenet_model.pth"
CLASS_MAP_PATH = "./class_idx_to_name.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms must match training
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load class mapping
with open(CLASS_MAP_PATH, "r") as f:
    idx2class = {int(k): v for k, v in json.load(f).items()}

# DenseNetHead used in training
class DenseNetHead(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes, dropout=0.6):
        super().__init__()
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

# Load backbone
try:
    backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
except:
    backbone = models.densenet121(pretrained=True)

feat_dim = backbone.classifier.in_features
model = DenseNetHead(backbone, feat_dim, num_classes=len(idx2class))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(DEVICE).eval()

app = FastAPI(title="Disaster Multimodal Inference API")

class PredictResponse(BaseModel):
    label: str
    class_idx: int
    confidence: float
    timestamp: float
    meta: dict

def preprocess_image(raw_bytes):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return eval_transform(img).unsqueeze(0)

@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    source: Optional[str] = Form(None),
    caption: Optional[str] = Form(None)
):
    start = time.time()
    raw = await image.read()
    x = preprocess_image(raw).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        label = idx2class[idx]

    return {
        "label": label,
        "class_idx": idx,
        "confidence": conf,
        "timestamp": time.time(),
        "meta": {
            "lat": lat,
            "lon": lon,
            "source": source,
            "caption": caption,
            "latency_s": round(time.time() - start, 3)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
