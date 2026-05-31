"""
inference/predict.py — BrailleVision Batch Inference & Evaluation Tool
=======================================================================
Usage examples:

  # Single image
  python inference/predict.py --source sample_inputs/braille1.jpg

  # Entire folder of images
  python inference/predict.py --source sample_inputs/

  # Video file
  python inference/predict.py --source tests/braille_video.mp4

  # Custom weights / thresholds
  python inference/predict.py --source sample_inputs/ --weights model/best.pt --conf 0.45

  # Export predictions to JSON
  python inference/predict.py --source sample_inputs/ --export-json

  # Use ONNX model (faster CPU inference)
  python inference/predict.py --source sample_inputs/ --weights model/best.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Sibling import: reuse shared logic from app.py ─────────────────────────────
# Insert the project root so we can import from app.py regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import (
    BrailleDetector,
    labels_to_text,
    sort_detections_reading_order,
    DEFAULT_CONF,
    DEFAULT_IOU,
    IMGSZ,
)

# ── Supported extensions ────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_WEIGHTS    = "model/best.pt"
DEFAULT_OUTPUT_DIR = "sample_outputs"


# ═══════════════════════════════════════════════════════════════════════════════
# Collect input files
# ═══════════════════════════════════════════════════════════════════════════════
def collect_inputs(source: str) -> tuple[list[Path], list[Path]]:
    """
    Return (image_paths, video_paths) from a file, folder, or glob pattern.
    Exits with a clear message if nothing is found.
    """
    src = Path(source)
    images, videos = [], []

    if src.is_file():
        ext = src.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(src)
        elif ext in VIDEO_EXTS:
            videos.append(src)
        else:
            sys.exit(f"[ERROR] Unsupported file extension: {ext}")
    elif src.is_dir():
        for f in sorted(src.iterdir()):
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(f)
            elif ext in VIDEO_EXTS:
                videos.append(f)
        if not images and not videos:
            sys.exit(f"[ERROR] No supported files found in directory: {src}")
    else:
        sys.exit(f"[ERROR] Source not found: {source}")

    return images, videos


# ═══════════════════════════════════════════════════════════════════════════════
# Per-image inference
# ═══════════════════════════════════════════════════════════════════════════════
def predict_image(
    image_path: Path,
    detector: BrailleDetector,
    output_dir: Path,
    save_annotated: bool = True,
) -> dict:
    """
    Run inference on one image.  Returns a result dict with:
        path, text, num_detections, labels, confidences, inference_ms
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  [SKIP] Cannot read: {image_path}")
        return {}

    t0 = time.perf_counter()
    results, text, dets = detector.predict_frame(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if save_annotated and dets:
        annotated = results[0].plot()
        out_path = output_dir / f"{image_path.stem}_result{image_path.suffix}"
        cv2.imwrite(str(out_path), annotated)
        saved_to = str(out_path)
    else:
        saved_to = None

    result = {
        "path":           str(image_path),
        "text":           text,
        "num_detections": len(dets),
        "labels":         [d["label"] for d in dets],
        "confidences":    [round(d["conf"], 4) for d in dets],
        "inference_ms":   round(elapsed_ms, 2),
        "saved_to":       saved_to,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Per-video inference
# ═══════════════════════════════════════════════════════════════════════════════
def predict_video(
    video_path: Path,
    detector: BrailleDetector,
    output_dir: Path,
    sample_every_n: int = 5,
) -> dict:
    """
    Run inference on a video, sampling every N frames.
    Saves an annotated output video and returns aggregated results.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open video: {video_path}")
        return {}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{video_path.stem}_result.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_idx      = 0
    all_labels:     list[str]  = []
    total_ms:       float      = 0.0
    processed      = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            t0 = time.perf_counter()
            results, _text, dets = detector.predict_frame(frame)
            total_ms += (time.perf_counter() - t0) * 1000
            last_annotated = results[0].plot()
            all_labels.extend(d["label"] for d in dets)
            processed += 1

        # Write the most-recent annotated frame (holds until next detection)
        writer.write(last_annotated if last_annotated is not None else frame)
        frame_idx += 1

    cap.release()
    writer.release()

    text = labels_to_text(all_labels)
    result = {
        "path":            str(video_path),
        "text":            text,
        "total_frames":    frame_idx,
        "frames_inferred": processed,
        "avg_inference_ms": round(total_ms / max(processed, 1), 2),
        "saved_to":        str(out_path),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty-print results
# ═══════════════════════════════════════════════════════════════════════════════
def print_results(results: list[dict]):
    total    = len(results)
    detected = sum(1 for r in results if r.get("text"))
    total_ms = sum(r.get("inference_ms", 0) for r in results)

    print("\n" + "═" * 60)
    print(f"  BrailleVision — Batch Inference Summary")
    print("═" * 60)

    for i, r in enumerate(results, 1):
        name   = Path(r["path"]).name
        text   = r.get("text") or "(none detected)"
        n_det  = r.get("num_detections", "—")
        ms     = r.get("inference_ms", "—")
        print(f"\n  [{i:02d}] {name}")
        print(f"       Text       : {text}")
        print(f"       Detections : {n_det}")
        print(f"       Time       : {ms} ms")
        if r.get("saved_to"):
            print(f"       Saved to   : {r['saved_to']}")

    print("\n" + "─" * 60)
    print(f"  Files processed  : {total}")
    print(f"  With detections  : {detected} / {total}")
    if total > 0:
        print(f"  Avg. infer. time : {total_ms / total:.1f} ms/frame")
    print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="BrailleVision batch inference / evaluation tool"
    )
    parser.add_argument("--source", required=True,
                        help="Image, video, or folder path")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"Model weights (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help=f"NMS IoU threshold (default: {DEFAULT_IOU})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for annotated outputs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving annotated output images/video")
    parser.add_argument("--export-json", action="store_true",
                        help="Export results to results.json in --output-dir")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="For videos: run inference every N frames (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = BrailleDetector(args.weights, args.conf, args.iou)
    images, videos = collect_inputs(args.source)

    print(f"\n[INFO] Found {len(images)} image(s) and {len(videos)} video(s) to process.\n")

    all_results: list[dict] = []

    # ── Process images ──────────────────────────────────────────────────────────
    for img_path in images:
        print(f"  Processing image: {img_path.name}", end=" ", flush=True)
        r = predict_image(img_path, detector, output_dir, save_annotated=not args.no_save)
        if r:
            all_results.append(r)
            status = r["text"] if r["text"] else "(no detection)"
            print(f"→ {status}")

    # ── Process videos ──────────────────────────────────────────────────────────
    for vid_path in videos:
        print(f"  Processing video: {vid_path.name} …")
        r = predict_video(vid_path, detector, output_dir, args.sample_every)
        if r:
            all_results.append(r)
            print(f"    → {r['text'] if r['text'] else '(no detection)'}")

    # ── Print summary ───────────────────────────────────────────────────────────
    if all_results:
        print_results(all_results)

    # ── Optional JSON export ────────────────────────────────────────────────────
    if args.export_json:
        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results exported to: {json_path}")


if __name__ == "__main__":
    main()