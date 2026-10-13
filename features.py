"""Feature extraction shared by training and prediction."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent
LABELS = ['stage_1', 'stage_2', 'stage_3', 'stage_4']
# Descriptive names follow the phase order on the publisher's dataset page.
# The original folder labels remain the authoritative prediction targets.
NAMES = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

def open_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')

def simple_features(image):
    a = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    hist = np.concatenate([np.histogram(a[:, :, c], bins=16, range=(0, 1))[0] / (224*224) for c in range(3)])
    cells = a.reshape(4, 56, 4, 56, 3).transpose(0, 2, 1, 3, 4)
    moments = np.concatenate([cells.mean((2, 3)).ravel(), cells.std((2, 3)).ravel()])
    gray = a.mean(2)
    gy, gx = np.gradient(gray)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    edges = np.histogram(angle, bins=9, range=(0, np.pi), weights=np.hypot(gx, gy))[0]
    edges /= max(float(edges.sum()), 1e-8)
    return np.concatenate([hist, moments, edges]).astype(np.float32)

def load_encoder():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    local = ROOT / 'models' / 'resnet18-imagenet.pth'
    weights = ResNet18_Weights.IMAGENET1K_V1
    if local.exists():
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(local, map_location='cpu', weights_only=True))
    else:
        torch.hub.set_dir(str(ROOT / '.cache' / 'torch'))
        model = resnet18(weights=weights)
        local.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), local)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms()

def encode_paths(paths, batch_size=24):
    model, transform = load_encoder()
    result = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = torch.stack([transform(open_rgb(p)) for p in paths[start:start+batch_size]])
            result.append(model(images).numpy())
            if start % (batch_size*10) == 0:
                print(f'ImageNet features: {min(start+batch_size, len(paths))}/{len(paths)}', flush=True)
    return np.concatenate(result)
