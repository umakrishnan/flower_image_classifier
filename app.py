"""
Tiny web UI for Oxford 102 flower classification (fine-tuned ViT).

Run from project root:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 (or PORT=3000 python app.py).
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from flask import Flask, Response, abort, render_template, request, send_from_directory, url_for
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
    preview_url = None

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
            preview_url = url_for("serve_upload", filename=filename)
            try:
                from inference import predict_upload

                with open(save_path, "rb") as f:
                    predictions = predict_upload(f, top_k=5)
            except Exception as exc:  # noqa: BLE001 — demo UI: show friendly message
                error = f"Could not classify image: {exc}"
                predictions = None

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        filename=filename,
        preview_url=preview_url,
    )


@app.route("/uploads/<filename>")
def serve_upload(filename: str):
    """Serve an uploaded file for preview (single basename only)."""
    safe = secure_filename(filename)
    if not safe or safe != filename or "/" in filename or "\\" in filename:
        abort(404)
    path = (UPLOAD_DIR / safe).resolve()
    if path.parent != UPLOAD_DIR.resolve() or not path.is_file():
        abort(404)
    mime, _ = mimetypes.guess_type(safe)
    return send_from_directory(UPLOAD_DIR, safe, mimetype=mime)


@app.route("/health")
def health():
    return {"ok": True}


@app.route("/ping")
def ping():
    """Plain text — if you see this in the browser, Flask is definitely responding."""
    return Response(
        "flower_classifier server OK\n",
        mimetype="text/plain; charset=utf-8",
    )


@app.route("/test-ui")
def test_ui():
    """Bright static page — if this is blank, the browser is not reaching Flask."""
    html = (
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
    return Response(html, mimetype="text/html; charset=utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # 0.0.0.0 avoids some localhost / port-forward quirks; still use 127.0.0.1 in the browser.
    host = os.environ.get("HOST", "0.0.0.0")
    use_reloader = os.environ.get("FLASK_NO_RELOADER", "").lower() not in ("1", "true", "yes")

    print(f"\n  Flower classifier — try in Chrome or Safari (not Cursor's preview):\n")
    print(f"    http://127.0.0.1:{port}/ping       ← should show one line of text")
    print(f"    http://127.0.0.1:{port}/test-ui    ← yellow page")
    print(f"    http://127.0.0.1:{port}/           ← main app\n")
    print("  When you refresh, this terminal should log a GET line. If it does not, the")
    print("  request never reached this process (wrong port, VPN, or different machine).\n")
    if not use_reloader:
        print("  (reloader disabled via FLASK_NO_RELOADER=1)\n")

    app.run(host=host, port=port, debug=True, use_reloader=use_reloader)
