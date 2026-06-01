"""
app.py — BrailleVision Real-Time Braille Reader
================================================
Modes:
  python app.py                  → real-time camera (webcam index 0)
  python app.py --source img.jpg → single image
  python app.py --source video.mp4 → video file
  python app.py --source 1       → webcam index 1
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── Optional TTS (graceful fallback if pyttsx3 missing) ────────────────────────
try:
    import pyttsx3
    _tts_available = True
except ImportError:
    _tts_available = False
    print("[WARN] pyttsx3 not found — speech output disabled. Install with: pip install pyttsx3")

from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS   = "model/best.pt"
DEFAULT_CONF      = 0.25          # lowered from 0.40 → catches low-confidence embossed letters
DEFAULT_IOU       = 0.35          # tighter NMS: prevents duplicate boxes on dense grids
IMGSZ             = 640
ROW_GAP_RATIO     = 0.55          # row gap = this × median box height (tuned for grid layouts)
SPEAK_COOLDOWN_S  = 3.0           # min seconds between auto-speech triggers
STABILITY_FRAMES  = 8             # frames the same text must hold before auto-speak
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ── Braille number indicator mapping ───────────────────────────────────────────
# When a '#' (number indicator) is detected, the next letters map to digits
_LETTER_TO_DIGIT = {
    'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
    'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0',
}


# ═══════════════════════════════════════════════════════════════════════════════
# TTS helper
# ═══════════════════════════════════════════════════════════════════════════════
class TTSEngine:
    """Thread-safe wrapper around pyttsx3 with a cooldown guard."""

    def __init__(self, rate: int = 160):
        self._engine = None
        self._last_spoken_at = 0.0
        if _tts_available:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
            except Exception as exc:
                print(f"[WARN] TTS init failed: {exc}")
                self._engine = None

    def speak(self, text: str, force: bool = False) -> bool:
        """
        Speak *text*.  Returns True if speech was triggered.
        Respects SPEAK_COOLDOWN_S unless force=True.
        """
        if not self._engine or not text.strip():
            return False
        now = time.time()
        if not force and (now - self._last_spoken_at) < SPEAK_COOLDOWN_S:
            return False
        self._last_spoken_at = now
        try:
            self._engine.say(f"Braille text: {text}")
            self._engine.runAndWait()
            return True
        except Exception as exc:
            print(f"[WARN] TTS speak failed: {exc}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Reading-order sort
# ═══════════════════════════════════════════════════════════════════════════════
def sort_detections_reading_order(boxes, names: dict) -> list[dict]:
    """
    Sort detections in natural reading order (top→bottom, left→right).

    Key fix over previous version:
      The old approach compared each detection to the *first* element of
      the current row.  On a grid chart (like an alphabet reference card),
      boxes in the same visual row can have cy values that drift >30 px from
      top to bottom of the row, so the group kept splitting mid-row.

      New approach — gap-based clustering:
        1. Sort all detections by cy (top to bottom).
        2. Look at the *gap* between consecutive cy values.
        3. A gap larger than ROW_GAP_RATIO × median_height signals a new row.
      This is robust to perspective tilt and varying cell heights because it
      reacts to *sudden jumps* rather than an absolute offset from row start.
    """
    if not boxes:
        return []

    dets = []
    heights = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h  = y2 - y1
        heights.append(h)
        dets.append({
            "cx": cx, "cy": cy,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "label": names[int(box.cls)],
            "conf": float(box.conf),
        })

    if len(dets) == 1:
        return dets

    # Threshold = fraction of median box height
    row_split_gap = max(15.0, float(np.median(heights)) * ROW_GAP_RATIO)

    # Sort by center-y
    dets.sort(key=lambda d: d["cy"])

    # Gap-based row splitting
    rows: list[list[dict]] = []
    current_row: list[dict] = [dets[0]]

    for prev, curr in zip(dets, dets[1:]):
        gap = curr["cy"] - prev["cy"]          # always ≥ 0 after sorting
        if gap > row_split_gap:
            rows.append(sorted(current_row, key=lambda d: d["cx"]))
            current_row = [curr]
        else:
            current_row.append(curr)
    rows.append(sorted(current_row, key=lambda d: d["cx"]))

    return [det for row in rows for det in row]


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing: raw label sequence → readable text
# ═══════════════════════════════════════════════════════════════════════════════
def labels_to_text(labels: list[str]) -> str:
    """
    Convert raw YOLO label sequence to human-readable text.

    Handles:
      • Capitalisation indicator  ('#' in some datasets maps to caps indicator)
      • Number indicator          ('#num' or a dedicated class)
      • Space characters
      • Unknown labels are passed through as-is
    """
    result = []
    capitalise_next = False
    number_mode = False

    for lbl in labels:
        lbl_lower = lbl.lower().strip()

        # Common special-class names used in Braille YOLO datasets
        if lbl_lower in ("space", " ", ""):
            result.append(" ")
            number_mode = False
            continue

        if lbl_lower in ("capital", "caps", "capital_indicator"):
            capitalise_next = True
            continue

        if lbl_lower in ("number", "num", "number_indicator", "#"):
            number_mode = True
            continue

        # Single letter
        char = lbl_lower
        if number_mode:
            char = _LETTER_TO_DIGIT.get(char, char)
            # Stay in number mode until a space
        elif capitalise_next:
            char = char.upper()
            capitalise_next = False

        result.append(char)

    return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Core detector wrapper
# ═══════════════════════════════════════════════════════════════════════════════
class BrailleDetector:
    """Wraps YOLO inference + reading-order sort + text assembly."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS,
                 conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU,
                 debug: bool = False):
        if not Path(weights).exists():
            sys.exit(f"[ERROR] Model weights not found: {weights}")
        print(f"[INFO] Loading model: {weights}")
        self.model = YOLO(weights)
        self.conf  = conf
        self.iou   = iou
        self.debug = debug
        print(f"[INFO] Model loaded  — conf={conf}  iou={iou}  debug={debug}")

    def predict_frame(self, frame: np.ndarray) -> tuple:
        """
        Run inference on a single BGR frame.

        Returns:
            results   — raw ultralytics Results object
            text      — assembled human-readable string
            dets      — sorted list of detection dicts (for drawing etc.)
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=IMGSZ,
            verbose=False,
        )
        dets   = []
        labels = []
        for r in results:
            if len(r.boxes) == 0:
                continue
            sorted_dets = sort_detections_reading_order(list(r.boxes), r.names)
            dets.extend(sorted_dets)
            labels.extend(d["label"] for d in sorted_dets)

        text = labels_to_text(labels)

        # Debug: print raw label sequence + row grouping so you can diagnose order issues
        if self.debug and labels:
            print(f"[DEBUG] Raw labels ({len(labels)}): {' '.join(labels)}")
            print(f"[DEBUG] Text output : {text!r}")

        return results, text, dets


# ═══════════════════════════════════════════════════════════════════════════════
# OSD (on-screen display) helpers
# ═══════════════════════════════════════════════════════════════════════════════
def draw_overlay(frame: np.ndarray, text: str, fps: float,
                 speaking: bool = False) -> np.ndarray:
    """Annotate frame with detected text, FPS, and status hints."""
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Detected text
    display_text = text if text else "— no Braille detected —"
    cv2.putText(frame, f"Braille: {display_text}", (12, 40),
                FONT, 0.9, (50, 255, 120), 2, cv2.LINE_AA)

    # FPS counter (top-right)
    fps_str = f"{fps:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_str, FONT, 0.6, 1)
    cv2.putText(frame, fps_str, (w - tw - 10, 22),
                FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Speaking indicator
    if speaking:
        cv2.putText(frame, "◉ SPEAKING", (12, h - 40),
                    FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    # Controls hint (bottom bar)
    hint = "  s = speak   r = reset   q = quit"
    cv2.putText(frame, hint, (12, h - 12),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Image mode
# ═══════════════════════════════════════════════════════════════════════════════
def run_on_image(source: str, detector: BrailleDetector,
                 tts: TTSEngine, output_dir: str = "sample_outputs") -> str:
    """Run detection on a single image file and save annotated output."""
    frame = cv2.imread(source)
    if frame is None:
        sys.exit(f"[ERROR] Cannot read image: {source}")

    results, text, _ = detector.predict_frame(frame)
    annotated = results[0].plot()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem      = Path(source).stem
    out_path  = str(Path(output_dir) / f"{stem}_braille_result.jpg")
    cv2.imwrite(out_path, annotated)

    print(f"\n{'─'*50}")
    print(f"  Source  : {source}")
    print(f"  Result  : {out_path}")
    print(f"  Braille : {text if text else '(none detected)'}")
    print(f"{'─'*50}\n")

    tts.speak(text, force=True)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Video / camera mode
# ═══════════════════════════════════════════════════════════════════════════════
def run_video(source, detector: BrailleDetector, tts: TTSEngine):
    """
    Real-time detection loop for webcam or video file.

    Key improvements:
      • FPS measurement via exponential moving average.
      • Stability buffer: text must persist for STABILITY_FRAMES before
        auto-speech (avoids babbling on flickering detections).
      • 'r' key clears history so you can reset to a new Braille cell.
      • Graceful camera-open failure with helpful message.
    """
    # Accept integer index or file path
    cam_index = source
    if isinstance(source, str) and source.isdigit():
        cam_index = int(source)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {source}")

    # Suggest a backend resolution for webcams
    if isinstance(cam_index, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[INFO] BrailleVision running!")
    print("       Controls:  s = speak now   r = reset history   q = quit\n")

    fps_ema        = 30.0          # exponential moving average FPS
    prev_time      = time.time()
    stability_buf  = deque(maxlen=STABILITY_FRAMES)   # recent text readings
    last_auto_text = ""
    speaking       = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or camera lost.")
            break

        results, text, _ = detector.predict_frame(frame)
        annotated = results[0].plot()

        # FPS
        now = time.time()
        dt  = max(now - prev_time, 1e-6)
        fps_ema    = 0.9 * fps_ema + 0.1 * (1.0 / dt)
        prev_time  = now

        # Stability check for auto-speech
        stability_buf.append(text)
        stable_text = text if (
            len(stability_buf) == STABILITY_FRAMES
            and len(set(stability_buf)) == 1     # all frames identical
            and text                              # non-empty
            and text != last_auto_text
        ) else ""

        speaking = False
        if stable_text:
            spoke = tts.speak(stable_text)
            if spoke:
                last_auto_text = stable_text
                speaking = True

        annotated = draw_overlay(annotated, text, fps_ema, speaking)
        cv2.imshow("BrailleVision", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tts.speak(text, force=True)
        elif key == ord('r'):
            stability_buf.clear()
            last_auto_text = ""
            print("[INFO] History reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] BrailleVision stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="BrailleVision — real-time Braille to text/speech"
    )
    parser.add_argument(
        "--source", default=None,
        help="Image/video path or webcam index (default: 0 = first webcam)"
    )
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS,
        help=f"Model weights path (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Detection confidence threshold (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU})"
    )
    parser.add_argument(
        "--no-speech", action="store_true",
        help="Disable text-to-speech output"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print raw detected labels + row grouping to terminal each frame"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    detector = BrailleDetector(args.weights, args.conf, args.iou, debug=args.debug)
    tts      = TTSEngine() if not args.no_speech else TTSEngine.__new__(TTSEngine)
    if args.no_speech:
        tts._engine = None   # disable TTS without needing a separate flag

    source = args.source

    # Determine mode
    if source is None:
        run_video(0, detector, tts)
    else:
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        if Path(source).suffix.lower() in img_exts and Path(source).exists():
            run_on_image(source, detector, tts)
        else:
            run_video(source, detector, tts)


if __name__ == "__main__":
    main()