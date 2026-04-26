"""Shared ResNet50 ImageNet inference for CLI and web UI."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

ModelBundle = Tuple[torch.nn.Module, transforms.Compose, List[str]]

_bundle: Optional[ModelBundle] = None


def _get_bundle() -> ModelBundle:
    global _bundle
    if _bundle is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.eval()
        categories = list(weights.meta["categories"])
        _bundle = (model, transform, categories)
    return _bundle


def predict_pil(image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
    """Return top-k (label, probability) pairs for an RGB PIL image."""
    model, transform, categories = _get_bundle()
    rgb = image.convert("RGB")
    tensor = transform(rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    k = max(1, min(top_k, probs.shape[0]))
    top_probs, top_indices = torch.topk(probs, k=k)

    return [
        (categories[idx.item()], float(prob.item()))
        for prob, idx in zip(top_probs, top_indices)
    ]


def predict_path(path: Path, top_k: int = 5) -> List[Tuple[str, float]]:
    with Image.open(path) as img:
        return predict_pil(img, top_k=top_k)


def predict_upload(file_obj: BinaryIO, top_k: int = 5) -> List[Tuple[str, float]]:
    data = file_obj.read()
    with Image.open(BytesIO(data)) as img:
        return predict_pil(img, top_k=top_k)
