"""
Tiny web UI for flower (ImageNet) classification with ResNet50.

Run from project root:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(_ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
UPLOAD_DIR = Path(__file__).resolve().parent / "images" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None
    filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "Choose an image file to upload."
        elif not allowed_file(file.filename):
            error = "Supported formats: PNG, JPG, JPEG, GIF, WEBP."
        else:
            filename = secure_filename(file.filename)
            save_path = UPLOAD_DIR / filename
            file.save(save_path)
            try:
                from inference import predict_upload

                with open(save_path, "rb") as f:
                    predictions = predict_upload(f, top_k=5)
            except Exception as exc:  # noqa: BLE001 — demo UI: show friendly message
                error = f"Could not read image: {exc}"
                predictions = None

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        filename=filename,
    )


@app.route("/health")
def health():
    return {"ok": True}


@app.route("/test-ui")
def test_ui():
    """Bright static page — if this is blank, the browser is not reaching Flask."""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Flask OK</title></head>"
        '<body style="margin:0;padding:2rem;background:#ffeb3b;color:#111;'
        'font-family:system-ui,sans-serif">'
        "<h1 style=\"margin-top:0\">Server reached Flask successfully.</h1>"
        "<p>If you see yellow and this text, networking is fine. "
        "Next open the real app:</p>"
        '<p><a href="/" style="font-size:1.2rem">Open flower classifier ( / )</a></p>'
        "</body></html>"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
