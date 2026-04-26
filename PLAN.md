# Flower classifier — workflow

**Remote:** [github.com/umakrishnan/flower_image_classifier](https://github.com/umakrishnan/flower_image_classifier)  
**Repo root:** this folder (`flower_classifier/`), not the parent workspace.

## Local setup (once per machine)

```bash
cd /path/to/flower_classifier
# e.g. conda: conda activate uvenv  OR  python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dirs.py
```

## Run the web UI (daily)

Default Flask port is **5000**, but on a Mac **AirPlay Receiver often uses 5000**, and **8080** is frequently taken by other tools. Use a free port explicitly:

```bash
PORT=3000 python app.py
```

Leave that terminal open. In **Chrome or Safari** (prefer not Cursor’s embedded preview if you see blank or “access denied”):

| URL | Purpose |
|-----|---------|
| `http://127.0.0.1:3000/ping` | Plain text — confirms the browser reached Flask |
| `http://127.0.0.1:3000/test-ui` | Bright yellow diagnostic page |
| `http://127.0.0.1:3000/` | Main classifier UI |

Use **`http://`**, not `https://`. If the Flask terminal shows **no `GET` line** when you refresh, the request is not hitting this process (wrong port, VPN, or wrong browser target).

Optional single-process mode if the reloader ever misbehaves:

```bash
FLASK_NO_RELOADER=1 PORT=3000 python app.py
```

## CLI classification

Put an image in `images/uploads/`, then:

```bash
python scripts/classify_flower.py --image images/uploads/your.jpg --top-k 5
```

## Git workflow

```bash
git pull
# edit…
git add .
git commit -m "Describe change"
git push origin main
```

Feature branches + PRs are recommended if you use **Bugbot** or CI (reviews run on PRs, not on every local commit).

## Demo note

The model is **Oxford 102 Flowers** (102 English flower names via a fine-tuned ViT). It is much more flower-specific than ImageNet, but wide field shots or odd cultivars can still be wrong—set expectations in live demos.
