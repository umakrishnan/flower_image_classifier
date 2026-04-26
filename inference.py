"""Oxford 102 flower classification via a fine-tuned ViT (Hugging Face)."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Apache-2.0, human-readable Oxford-102 class names in config.id2label
DEFAULT_MODEL_ID = "loretyan/vit-base-oxford-flowers-102"

Bundle = Tuple[AutoImageProcessor, AutoModelForImageClassification, Dict[int, str]]

_bundle: Optional[Bundle] = None


def _normalize_id2label(raw: dict) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k, v in raw.items():
        idx = int(k) if not isinstance(k, int) else k
        out[idx] = str(v)
    return out


def _get_bundle() -> Bundle:
    global _bundle
    if _bundle is None:
        model_id = os.environ.get("FLOWER_MODEL_ID", DEFAULT_MODEL_ID)
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.eval()
        id2label = _normalize_id2label(dict(model.config.id2label))
        _bundle = (processor, model, id2label)
    return _bundle


def predict_pil(image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Return top-k (label, probability) for an RGB-ish PIL image.
    Images are resized and normalized by the ViT processor (typically 224×224).
    """
    processor, model, id2label = _get_bundle()
    rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0], dim=0)

    k = max(1, min(top_k, probs.shape[0]))
    top_probs, top_indices = torch.topk(probs, k=k)

    return [
        (id2label[int(idx.item())], float(prob.item()))
        for prob, idx in zip(top_probs, top_indices)
    ]


def predict_path(path: Path, top_k: int = 5) -> List[Tuple[str, float]]:
    with Image.open(path) as img:
        return predict_pil(img, top_k=top_k)


def predict_upload(file_obj: BinaryIO, top_k: int = 5) -> List[Tuple[str, float]]:
    data = file_obj.read()
    with Image.open(BytesIO(data)) as img:
        return predict_pil(img, top_k=top_k)
