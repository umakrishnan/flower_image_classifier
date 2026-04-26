#!/usr/bin/env python3
"""Create expected project directories if they do not exist."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folders = [
        root / "images" / "uploads",
        root / "images" / "samples",
        root / "models",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {folder}")


if __name__ == "__main__":
    main()
