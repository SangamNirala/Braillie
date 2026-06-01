# BrailleVision 👁️ — Real-Time Braille to English

> An assistive AI system that reads real physical/embossed Braille using a camera and converts it into English text and speech in real time.

---

## 📌 Project Overview

BrailleVision is a deep learning-based assistive technology built for the **BrailleVision Hackathon 2026**. It uses a fine-tuned YOLOv8 object detection model to detect and classify all 26 Braille letters (A–Z) from camera input or uploaded images, then assembles them into readable English text and optionally speaks it aloud.

The system successfully detects **all 26 Braille letters (A–Z) in a single image** with an average confidence of 85% and achieves **98.0% mAP@0.5** through a two-phase transfer learning training strategy.

---

## 🎯 Key Features

- 📷 **Real-time camera detection** — point a webcam at physical Braille and see live results
- 🖼️ **Web app upload mode** — drag and drop a Braille photo, get instant annotated output
- 🔠 **Braille dot visualizer** — scrollable visual display of each detected Braille cell
- 🔊 **Text-to-speech** — speak button reads detected text aloud via browser Speech API
- 📋 **Copy to clipboard** — one-click copy of detected text
- 📊 **Per-letter confidence** — shows confidence % for every detected character
- 🌐 **FastAPI web app** — full browser-based UI, no installation needed for judges
- ⚡ **~30ms inference** — real-time performance on GPU
- 🔤 **Reading-order sorting** — characters assembled left-to-right, top-to-bottom

---

## 📊 Model Performance

### Final Model (Phase 2 — used in production)

| Metric | Score |
|---|---|
| **mAP@0.5** | **0.980 (98.0%)** |
| **mAP@0.5:0.95** | **0.771 (77.1%)** |
| **Precision** | **0.943 (94.3%)** |
| **Recall** | **0.976 (97.6%)** |
| **Inference Speed** | **~24ms per image (GPU)** |
| **Total Training Time** | **3.6 hours (Tesla T4 GPU)** |

### Phase Comparison

| Phase | mAP@0.5 | mAP@0.5:0.95 | Epochs | Duration |
|---|---|---|---|---|
| Phase 1 (heavy augmentation) | 0.978 | 0.756 | 120 | 2.6 hrs |
| **Phase 2 (fine-tuning)** | **0.980** | **0.771** | 40 | 1.0 hr |

### Per-Class AP@0.5 (Final Model)

| Letter | AP@0.5 | Letter | AP@0.5 |
|---|---|---|---|
| A | 0.971 | N | 0.991 |
| B | 0.967 | O | 0.992 |
| C | 0.993 | P | 0.990 |
| D | 0.987 | Q | **0.995** |
| E | 0.993 | R | 0.986 |
| F | 0.941 ⚠️ | S | 0.991 |
| G | **0.995** | T | 0.986 |
| H | 0.990 | U | **0.995** |
| I | 0.990 | V | 0.992 |
| J | 0.885 ⚠️ | W | 0.988 |
| K | 0.968 | X | 0.970 |
| L | 0.983 | Y | 0.961 |
| M | 0.978 | Z | **0.995** |

> ⚠️ J and F have slightly lower AP due to fewer training samples in the dataset — all other letters are above 0.94.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 medium (Ultralytics 8.4.58) |
| Training Framework | PyTorch + CUDA 12.8 |
| Computer Vision | OpenCV |
| Text-to-Speech | Web Speech API (browser) / pyttsx3 (CLI) |
| Web Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JavaScript |
| Training Platform | Google Colab (Tesla T4 16GB GPU) |
| Dataset Platform | Roboflow |
| Model Export | ONNX (for deployment) |

---

## 📂 Repository Structure

```
braillevision/
│
├── app.py                      # Main app — camera, image, video modes + TTS
├── main.py                     # FastAPI web application backend
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── setup_instructions.md       # Detailed setup guide
├── ai_tools_disclosure.md      # Tools and references used
│
├── static/
│   └── index.html              # Web UI — drag/drop upload, Braille visualizer
│
├── model/
│   ├── yolov8_braille.pt       # Base pretrained Braille weights (DotNeuralNet)
│   ├── best.pt                 # Our fine-tuned model weights (52.1 MB)
│   └── best.onnx               # ONNX export for deployment (99.1 MB)
│
├── training/
│   ├── train.py                # Two-phase training script (v4)
│   └── results/
│       ├── phase1/             # Phase 1 results (120 epochs)
│       │   ├── results.png
│       │   └── confusion_matrix.png
│       └── phase2/             # Phase 2 results (40 epochs) — FINAL
│           ├── results.png
│           ├── confusion_matrix.png
│           └── training_report.txt
│
├── dataset/
│   ├── data.yaml               # Dataset config (26 classes A-Z)
│   ├── train/                  # 1,757 training images + labels
│   ├── valid/                  # 206 validation images + labels
│   ├── test/                   # Test images
│   └── dataset_info.md
│
├── inference/
│   └── predict.py              # Batch inference + JSON export tool
│
├── sample_inputs/              # Sample Braille images for judges to test
├── sample_outputs/             # Annotated output images + results.json
│
└── demo/
    └── demo_video_link.txt
```

---

## 🗃️ Dataset

| Property | Details |
|---|---|
| **Name** | braillify |
| **Source** | Roboflow Universe |
| **Link** | https://universe.roboflow.com/nicco-van-hamja-b1vxy/braillify |
| **Total Images** | 2,062 real physical Braille images |
| **Classes** | 26 (A–Z) |
| **Format** | YOLOv8 PyTorch (bounding box annotations) |
| **Train Split** | 1,757 images |
| **Val Split** | 206 images |
| **Test Split** | Separate test folder |

---

## 🤖 Model & Training Details

### Architecture
- **Base model:** YOLOv8m — 25.8M parameters, 79.1 GFLOPs, 93 layers (fused)
- **Transfer learning from:** `yolov8_braille.pt` (DotNeuralNet — already Braille-trained)
- **Fine-tuned on:** braillify Roboflow dataset
- **Output classes:** 26 (A–Z Braille letters)
- **Recommended inference size:** 640×640 (800 showed no improvement)

### Two-Phase Training Strategy

```
Phase 1 — Heavy Augmentation (120 epochs)
─────────────────────────────────────────
Goal      : Learn all Braille dot patterns robustly
Optimizer : AdamW  lr=0.0005  → cosine decay
Batch     : Auto (6 for T4 GPU)
Key augs  : mosaic=1.0, mixup=0.15, copy_paste=0.15
            scale=0.5, erasing=0.3
            flipud=0 (Braille is orientation-sensitive)
Patience  : 50 epochs early stopping
Result    : mAP@0.5 = 0.978

Phase 2 — Fine-Tuning (40 epochs)
──────────────────────────────────
Goal      : Clean convergence on true data distribution
Optimizer : AdamW  lr=0.00005 (10× lower)  → cosine decay
Batch     : Auto
Key augs  : Minimal — mosaic=0, mixup=0, copy_paste=0
            Small rotation/scale only
Patience  : 20 epochs early stopping
Result    : mAP@0.5 = 0.980  (+0.002 over Phase 1)
```

### Training Script Features (train.py v4)

- ✅ **Google Drive backup** — saves weights every 10 epochs, survives Colab disconnects
- ✅ **Resume from checkpoint** — `--resume` flag restores from Drive if local files lost
- ✅ **Custom epoch logging** — clean one-line-per-epoch output (no per-batch spam)
- ✅ **Per-class AP analysis** — identifies weak letters automatically
- ✅ **imgsz comparison** — evaluates at both 640 and 800 and recommends best
- ✅ **ONNX export** — auto-exports final model after training
- ✅ **Training report** — saves JSON + TXT report to Drive and local

### Training Commands

```bash
# Standard two-phase training
python train.py

# Resume after Colab disconnect
python train.py --resume

# Skip phase 1, fine-tune from existing phase 1 weights
python train.py --phase2-only

# Evaluate only (no retraining)
python train.py --eval-only

# Disable early stopping
python train.py --no-early-stop

# Higher resolution
python train.py --imgsz 800
```

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.9 or above
- pip
- Webcam (for real-time camera mode)

### Install

```bash
pip install -r requirements.txt
```

### requirements.txt

```
ultralytics==8.4.58
torch
torchvision
opencv-python-headless
pyttsx3
numpy
Pillow
fastapi
uvicorn
python-multipart
```

---

## 🚀 How to Run

### Option 1 — Web App (Recommended for Demo)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open browser at: **http://localhost:8000**

**Web app features:**
- Drag and drop or click to upload Braille image
- Scrollable Braille dot visualizer for each detected letter
- Side-by-side original vs annotated image
- Detected text in large font with character count and inference time
- Average confidence meter
- 🔊 Speak button — reads detected text aloud
- 📋 Copy button — copies text to clipboard
- ↺ Reset button — clear and try another image

---

### Option 2 — Real-time Camera

```bash
python app.py
```

Controls: `s` = speak | `r` = reset | `q` = quit

---

### Option 3 — Single Image

```bash
python app.py --source sample_inputs/test1.jpg

# Without speech (headless/Codespaces)
python app.py --source sample_inputs/test1.jpg --no-speech
```

---

### Option 4 — Batch Test (for judges)

```bash
# Test all sample images
python inference/predict.py --source sample_inputs/

# Export detailed JSON results
python inference/predict.py --source sample_inputs/ --export-json

# Use ONNX model (faster CPU inference)
python inference/predict.py --source sample_inputs/ --weights model/best.onnx
```

---

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `None` (webcam 0) | Image / video path or webcam index |
| `--weights` | `model/best.pt` | Path to model weights |
| `--conf` | `0.50` | Detection confidence threshold |
| `--iou` | `0.40` | NMS IoU threshold |
| `--no-speech` | `False` | Disable text-to-speech |

---

## 🌐 FastAPI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/predict` | POST | Accepts image upload, returns detections |
| `/health` | GET | Health check |

### `/predict` Response Example

```json
{
  "success": true,
  "text": "abcdefghijklmnopqrstuvwxyz",
  "letters": [
    {"letter": "A", "confidence": 86.0},
    {"letter": "B", "confidence": 84.0},
    {"letter": "C", "confidence": 93.0}
  ],
  "count": 26,
  "inference_ms": 24.2,
  "image_b64": "..."
}
```

---

## 🔍 How It Works

```
Physical Braille input (camera / uploaded image)
                  ↓
      OpenCV frame capture / decode
                  ↓
      YOLOv8 detection
      (detects each Braille cell, classifies as A–Z)
                  ↓
      Reading-order sort
      (dynamic row grouping — left→right, top→bottom)
                  ↓
      Label sequence → English text assembly
                  ↓
      Web UI display + Braille dot visualizer + TTS
```

---

## 🧠 Transfer Learning Strategy

```
Stage 1: yolov8m.pt
         (COCO pretrained — general object detection)
              ↓  [DotNeuralNet training on Braille]
Stage 2: yolov8_braille.pt
         (Braille-aware base — knows dot patterns)
              ↓  [Phase 1: Our heavy augmentation training — 120 epochs]
Stage 3: braille_v4_phase1/best.pt
         (mAP@0.5 = 0.978)
              ↓  [Phase 2: Our fine-tuning — 40 epochs, 10× lower LR]
Stage 4: braille_v4_phase2/best.pt  ← FINAL MODEL
         (mAP@0.5 = 0.980)
```

This approach required only **3.6 hours total** to achieve research-grade accuracy across all 26 Braille letters.

---

## 📜 References & Credits

| Resource | Link |
|---|---|
| DotNeuralNet (base weights) | https://github.com/snoop2head/DotNeuralNet |
| braillify dataset (Roboflow) | https://universe.roboflow.com/nicco-van-hamja-b1vxy/braillify |
| Ultralytics YOLOv8 | https://github.com/ultralytics/ultralytics |
| FastAPI | https://fastapi.tiangolo.com |

---

## 🤝 AI Tools Disclosure

- **Claude (Anthropic)** — project guidance, code architecture, training pipeline design
- **DotNeuralNet** — pretrained `yolov8_braille.pt` used as base weights for transfer learning
- **Roboflow** — dataset hosting and YOLOv8 format export
- **Google Colab** — Tesla T4 GPU used for model training

---

## ✅ Submission Checklist

- [x] Public GitHub repository with complete source code
- [x] `README.md` with setup and run instructions
- [x] `requirements.txt`
- [x] `dataset/data.yaml` with class names and paths
- [x] Training code (`training/train.py` — two-phase v4)
- [x] Inference code (`inference/predict.py`, `app.py`)
- [x] Model weights (`model/best.pt`, `model/best.onnx`)
- [x] Training results — loss curves + confusion matrix (Phase 1 & Phase 2)
- [x] Training report (`training_report.txt`)
- [x] Sample inputs and annotated outputs
- [x] FastAPI web application with Braille dot visualizer
- [x] AI tools disclosure
- [x] Demo video

---

*BrailleVision — Making Braille accessible through AI | mAP 98.0% | All 26 letters detected*
