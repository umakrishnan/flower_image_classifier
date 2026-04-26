#!/usr/bin/env python3
"""
Classify a flower image using a pre-trained ResNet model.

Usage:
    python scripts/classify_flower.py --image images/uploads/flower.jpg --top-k 5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import predict_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flower image classification with pre-trained ResNet50."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the flower image to classify.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("Loading pre-trained ResNet50 weights (first run may download)...")
    top_k = max(1, args.top_k)
    rows = predict_path(image_path, top_k=top_k)

    print(f"\nTop {len(rows)} predictions for {image_path}:")
    for rank, (label, prob) in enumerate(rows, start=1):
        print(f"{rank}. {label:30s} {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
