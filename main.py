"""
main.py — BrailleVision FastAPI (with live camera support)
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import BrailleDetector

# ── Init ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BrailleVision API")

try:
    # Uses same conf=0.25 as app.py — no extra modifications
    detector = BrailleDetector(weights="model/best.pt", conf=0.25)
except SystemExit as e:
    print(f"[ERROR] Model load failed: {e}")
    detector = None

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Helper: decode uploaded bytes to cv2 frame ────────────────────────────────
def decode_image(contents: bytes) -> np.ndarray:
    np_arr = np.frombuffer(contents, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image.")
    return frame


# ── Helper: run YOLO inference on a frame ─────────────────────────────────────
def run_inference(frame: np.ndarray) -> dict:
    if detector is None:
        raise HTTPException(500, "Model not loaded — check model/best.pt exists.")
    try:
        t0 = time.perf_counter()
        results, text, dets = detector.predict_frame(frame)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

    annotated = results[0].plot()
    _, buffer  = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64    = base64.b64encode(buffer).decode("utf-8")

    letters = [
        {"letter": d["label"], "confidence": round(d["conf"] * 100, 1)}
        for d in dets
    ]

    return {
        "success":      True,
        "text":         text if text else "No Braille detected",
        "letters":      letters,
        "count":        len(dets),
        "inference_ms": elapsed_ms,
        "image_b64":    img_b64,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Static image upload — full detection response with per-letter breakdown."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")
    frame = decode_image(await file.read())
    return JSONResponse(run_inference(frame))


@app.post("/predict_frame")
async def predict_frame_endpoint(file: UploadFile = File(...)):
    """
    Live webcam frame endpoint.
    Called repeatedly by the browser during camera mode.
    Returns a lighter response (no per-letter breakdown) to keep latency low.
    """
    frame = decode_image(await file.read())
    data  = run_inference(frame)
    return JSONResponse({
        "success":      data["success"],
        "text":         data["text"],
        "count":        data["count"],
        "inference_ms": data["inference_ms"],
        "image_b64":    data["image_b64"],
    })