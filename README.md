# Flower Classifier (ResNet50)

This project classifies an uploaded flower image using a pre-trained ResNet50 model from `torchvision`.

## Project structure

```
flower_classifier/
├── app.py              # Tiny Flask web UI
├── inference.py        # Shared ResNet50 inference
├── templates/
│   └── index.html
├── images/
│   ├── uploads/        # Put uploaded images here (web UI saves here too)
│   └── samples/        # Optional sample images
├── models/             # Reserved for local/custom model files
├── scripts/
│   ├── classify_flower.py
│   └── prepare_dirs.py
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure directories exist:

   ```bash
   python scripts/prepare_dirs.py
   ```

## Run classification

1. Place an image file in `images/uploads/` (for example `my_flower.jpg`).
2. Run:

   ```bash
   python scripts/classify_flower.py --image images/uploads/my_flower.jpg --top-k 5
   ```

The script prints the top ImageNet class predictions and confidence scores.

## Web UI (demo)

From the project directory:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), upload a flower photo, and click **Classify**. The first request downloads ResNet50 weights; allow network access for that step.

Optional: `PORT=8080 python app.py` to listen on another port.
