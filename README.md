# BrailleVision 👁️ — Real-Time Braille to English

> An assistive AI system that reads real physical/embossed Braille using a camera and converts it into English text and speech in real time.

---

## 📌 Project Overview

BrailleVision is a deep learning-based assistive technology built for the **BrailleVision Hackathon 2026**. It uses a fine-tuned YOLOv8 object detection model to detect and classify all 26 Braille letters (A–Z) from camera input or uploaded images, then assembles them into readable English text and optionally speaks it aloud.

---

## 🎯 Key Features

- 📷 **Real-time camera detection** — point a webcam at physical Braille and see live results
- 🖼️ **Image upload mode** — upload a Braille photo via web app or CLI
- 🔊 **Text-to-speech** — detected text is spoken aloud using pyttsx3 or browser TTS
- 🌐 **FastAPI web app** — browser-based UI with drag-and-drop upload, side-by-side comparison, and per-letter confidence breakdown
- 📊 **98.1% mAP@0.5** — state-of-the-art accuracy on 26 Braille classes
- 🔤 **Reading-order sorting** — characters assembled left-to-right, top-to-bottom
- ⚡ **30ms inference** — real-time performance on GPU

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| mAP@0.5 | **0.981 (98.1%)** |
| mAP@0.5:0.95 | **0.764 (76.4%)** |
| Precision | **0.950 (95.0%)** |
| Recall | **0.963 (96.3%)** |
| Inference Speed | **~30ms per image** |
| Training Time | **2.2 hours (100 epochs, Tesla T4 GPU)** |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics 8.4.58) |
| Training Framework | PyTorch + CUDA |
| Computer Vision | OpenCV |
| Text-to-Speech | pyttsx3 / Web Speech API |
| Web Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Training Platform | Google Colab (Tesla T4 GPU) |
| Dataset Platform | Roboflow |

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
│   └── index.html              # Web UI frontend
│
├── model/
│   ├── yolov8_braille.pt       # Base pretrained Braille weights (DotNeuralNet)
│   ├── best.pt                 # Our fine-tuned model weights (52.1 MB)
│   └── best.onnx               # ONNX export for deployment (99.6 MB)
│
├── training/
│   ├── train.py                # Training script
│   └── results/
│       ├── results.png         # Loss and mAP curves
│       ├── confusion_matrix.png
│       └── confusion_matrix_normalized.png
│
├── dataset/
│   ├── data.yaml               # Dataset config (26 classes, paths)
│   ├── train/                  # 1,757 training images + labels
│   ├── valid/                  # 206 validation images + labels
│   ├── test/                   # Test images
│   └── dataset_info.md         # Dataset documentation
│
├── inference/
│   └── predict.py              # Batch inference + JSON export tool
│
├── sample_inputs/              # Sample Braille images for testing
├── sample_outputs/             # Annotated output images + results.json
│
└── demo/
    └── demo_video_link.txt     # Link to demo video
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
| **Annotation** | Bounding boxes around individual Braille cells |

---

## 🤖 Model Details

### Architecture
- **Base model:** YOLOv8m (medium) — 25.8M parameters, 79.1 GFLOPs
- **Transfer learning from:** `yolov8_braille.pt` (DotNeuralNet — already Braille-trained)
- **Fine-tuned on:** braillify Roboflow dataset
- **Output classes:** 26 (A–Z Braille letters)

### Training Configuration

```
Epochs        : 100 (early stopping patience = 25)
Image size    : 640 × 640
Batch size    : 6 (auto-selected for Tesla T4 GPU)
Optimizer     : AdamW  lr=0.0005
LR Schedule   : Cosine decay
AMP           : Enabled (mixed precision)
Cache         : RAM
Augmentations : Horizontal flip, rotation ±5°, scale, mosaic,
                mixup=0.1, copy_paste=0.1
                (vertical flip DISABLED — Braille is orientation-sensitive)
```

### Training Results

| Epoch | mAP@0.5 |
|---|---|
| 1 | 0.681 |
| 5 | 0.897 |
| 14 | 0.953 |
| 32 | 0.971 |
| 65 | 0.983 |
| 100 | **0.981** |

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.9 or above
- pip
- Webcam (for real-time mode)

### Install dependencies

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

Features:
- Drag and drop Braille image upload
- Side-by-side original vs detected result
- Per-letter confidence breakdown
- Speak detected text button
- Download annotated result button

---

### Option 2 — Real-time Camera

```bash
python app.py
```

Controls:
- `s` → speak detected text
- `r` → reset detection history
- `q` → quit

---

### Option 3 — Single Image

```bash
python app.py --source sample_inputs/test1.jpg
```

Without speech (headless servers / Codespaces):

```bash
python app.py --source sample_inputs/test1.jpg --no-speech
```

---

### Option 4 — Batch Test (for judges)

```bash
# Test all images in sample_inputs/
python inference/predict.py --source sample_inputs/

# Export detailed JSON results
python inference/predict.py --source sample_inputs/ --export-json

# Use ONNX model (faster CPU inference)
python inference/predict.py --source sample_inputs/ --weights model/best.onnx
```

---

### Option 5 — Custom Parameters

```bash
python app.py --weights model/best.pt --conf 0.45 --iou 0.40
```

| Argument | Default | Description |
|---|---|---|
| `--source` | `None` (webcam) | Image / video path or webcam index |
| `--weights` | `model/best.pt` | Path to model weights |
| `--conf` | `0.50` | Detection confidence threshold |
| `--iou` | `0.40` | NMS IoU threshold |
| `--no-speech` | `False` | Disable text-to-speech |

---

## 🧪 Testing the Model

### Quick test on sample image:

```bash
python inference/predict.py --source sample_inputs/test1.jpg
```

### Expected output:

```
══════════════════════════════════════════════════════════
  BrailleVision — Batch Inference Summary
══════════════════════════════════════════════════════════

  [01] test1.jpg
       Text       : HELLO
       Detections : 5
       Time       : 30.4 ms
       Saved to   : sample_outputs/test1_result.jpg

──────────────────────────────────────────────────────────
  Files processed  : 1
  With detections  : 1 / 1
  Avg. infer. time : 30.4 ms/frame
══════════════════════════════════════════════════════════
```

---

## 🌐 FastAPI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/predict` | POST | Accepts image upload, returns detections |
| `/health` | GET | Health check |

### `/predict` Response Example:

```json
{
  "success": true,
  "text": "HELLO",
  "letters": [
    {"letter": "H", "confidence": 96.2},
    {"letter": "E", "confidence": 97.8},
    {"letter": "L", "confidence": 94.1},
    {"letter": "L", "confidence": 95.3},
    {"letter": "O", "confidence": 98.1}
  ],
  "count": 5,
  "inference_ms": 30.4,
  "image_b64": "..."
}
```

---

## 🔍 How It Works

```
Physical Braille input (camera / image)
              ↓
    OpenCV frame capture
              ↓
    YOLOv8 detection
    (detects each Braille cell as A–Z)
              ↓
    Reading-order sort
    (left→right, top→bottom using dynamic row grouping)
              ↓
    Label sequence → English text assembly
              ↓
    Display on screen + Text-to-speech output
```

---

## 📈 Per-Class Accuracy (Validation Set)

| Letter | mAP@0.5 | Letter | mAP@0.5 |
|---|---|---|---|
| A | 0.970 | N | 0.989 |
| B | 0.957 | O | 0.991 |
| C | 0.992 | P | 0.993 |
| D | 0.990 | Q | **0.995** |
| E | 0.990 | R | 0.990 |
| F | 0.951 | S | 0.986 |
| G | 0.980 | T | 0.985 |
| H | 0.990 | U | 0.989 |
| I | 0.985 | V | 0.991 |
| J | 0.924 | W | 0.974 |
| K | 0.986 | X | 0.978 |
| L | 0.982 | Y | 0.977 |
| M | 0.979 | Z | 0.986 |

---

## 🧠 Transfer Learning Approach

Rather than training from scratch, we used a two-stage transfer learning strategy:

```
Stage 1: yolov8m.pt (COCO pretrained — general object detection)
              ↓  [DotNeuralNet training]
Stage 2: yolov8_braille.pt (Braille-aware base weights)
              ↓  [Our fine-tuning on braillify dataset]
Stage 3: best.pt (Our final model — 98.1% mAP)
```

This approach required only **100 epochs** and **2.2 hours** to achieve research-grade accuracy, compared to 500+ epochs from scratch.

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

- **Claude (Anthropic)** — project guidance, code architecture, training pipeline
- **DotNeuralNet** — pretrained `yolov8_braille.pt` used as base weights for transfer learning
- **Roboflow** — dataset hosting and YOLOv8 format export
- **Google Colab** — Tesla T4 GPU used for model training

---

## ✅ Submission Checklist

- [x] Public GitHub repository with complete source code
- [x] `README.md` with setup and run instructions
- [x] `requirements.txt`
- [x] `dataset/data.yaml` with class names and paths
- [x] Training code (`training/train.py`)
- [x] Inference code (`inference/predict.py`, `app.py`)
- [x] Model weights (`model/best.pt`, `model/best.onnx`)
- [x] Training results (loss curves, confusion matrix)
- [x] Sample inputs and outputs
- [x] FastAPI web application
- [x] AI tools disclosure
- [x] Demo video

---

*BrailleVision — Making Braille accessible through AI*
