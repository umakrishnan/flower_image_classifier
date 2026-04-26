# Flower classifier (Oxford 102)

This project classifies an uploaded flower image using a **Vision Transformer (ViT)** fine-tuned on the **Oxford 102 Flowers** dataset, served from [Hugging Face](https://huggingface.co/loretyan/vit-base-oxford-flowers-102) (Apache-2.0). It is **much more flower-specific** than a generic ImageNet ResNet.

**Model:** `loretyan/vit-base-oxford-flowers-102` — 102 English flower category names (tulip, rose, sunflower, …). Override with env `FLOWER_MODEL_ID` if you swap checkpoints.

**Image size:** Any reasonable photo or screenshot works. The Hugging Face **image processor** resizes and normalizes to **224×224** before inference (you do not need to resize files yourself).

## Project structure

```
flower_classifier/
├── app.py              # Tiny Flask web UI
├── inference.py        # Shared ViT inference
├── templates/
│   └── index.html
├── images/
│   ├── uploads/        # Put uploaded images here (web UI saves here too)
│   └── samples/        # Optional sample images
├── models/             # Reserved for local/custom model files
├── scripts/
│   ├── classify_flower.py
│   └── prepare_dirs.py
├── PLAN.md             # Day-to-day workflow notes
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

## Run classification (CLI)

1. Place an image file in `images/uploads/` (for example `my_flower.jpg`).
2. Run:

   ```bash
   python scripts/classify_flower.py --image images/uploads/my_flower.jpg --top-k 5
   ```

The script prints the top **Oxford 102** predictions and confidence scores.

## Web UI (demo)

From the project directory:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), upload a flower photo, and click **Classify**. The UI shows a **preview** of the selected file (before submit) and the **uploaded image** after classification. The first run downloads ViT weights from Hugging Face; allow network access for that step.

On a Mac, port **5000** is often taken by AirPlay; **8080** is often busy. Example:

```bash
PORT=3000 python app.py
```

Then open `http://127.0.0.1:3000/`.

### Blank page in the browser?

1. Use **Chrome or Safari** at `http://127.0.0.1:5000` (must be **http**, not https).
2. Open `http://127.0.0.1:5000/ping` — you should see plain text `flower_classifier server OK`.
3. While refreshing, watch the terminal running `python app.py`: you should see a **GET** log line. **No log line** means nothing reached this app (wrong port, VPN, or Cursor preview not using your machine’s localhost).
4. If the terminal shows GETs but the tab stays white, try `FLASK_NO_RELOADER=1 python app.py` (single process, avoids rare reloader quirks).

## Limitations

- **102 categories only** — unusual cultivars or “flowers in the wild” far from training may still be mislabeled.
- **Not medical / production** — demo and learning use.
