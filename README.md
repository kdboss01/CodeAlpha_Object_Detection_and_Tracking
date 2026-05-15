# 🎯 Real-Time Object Detection & Tracking

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green?style=for-the-badge&logo=opencv)
![Deep SORT](https://img.shields.io/badge/Tracker-Deep%20SORT-purple?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia)

A real-time object detection and tracking system built with **YOLOv8** and **Deep SORT**.  
Detects objects across video frames and assigns persistent unique IDs to each tracked entity.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)

---

## 🔍 Overview

This project implements a **real-time object detection and multi-object tracking (MOT)** pipeline using:

- **YOLOv8** (You Only Look Once v8) as the detection backbone — fast, accurate, pretrained on COCO (80 classes)
- **Deep SORT** (Simple Online and Realtime Tracking with a Deep Association Metric) as the tracking algorithm — assigns and maintains unique IDs per object across frames

The system can process live webcam feeds or pre-recorded video files, annotating each frame with bounding boxes, class labels, and persistent tracking IDs in real time.

---

## ✨ Features

- 🎥 **Dual input support** — webcam (live) or video file
- 🧠 **YOLOv8 detection** — pretrained on 80 COCO classes (person, car, dog, etc.)
- 🔁 **Deep SORT tracking** — persistent IDs with appearance-based re-identification
- 🎨 **Per-class color coding** — visually distinct colors for each object class
- 📊 **Live HUD overlay** — real-time FPS counter and active track count
- 📸 **Screenshot capture** — press `s` anytime to save the current frame
- 💾 **Video output** — optional save of the annotated output as `.mp4`
- ⚡ **GPU acceleration** — CUDA + FP16 (half precision) support for RTX GPUs
- 🔧 **Single-file config** — all tunable parameters at the top of the script

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT SOURCE                         │
│              Webcam (cv2.VideoCapture(0))                   │
│           or Video File (cv2.VideoCapture(path))            │
└───────────────────────────┬─────────────────────────────────┘
                            │  raw BGR frame
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   OBJECT DETECTION (YOLOv8)                 │
│                                                             │
│  • Runs every FRAME_SKIP frames                             │
│  • Returns: bounding boxes (xyxy), confidence, class ID     │
│  • Filters detections below CONF_THRESHOLD                  │
└───────────────────────────┬─────────────────────────────────┘
                            │  detections: ([x,y,w,h], conf, cls)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  OBJECT TRACKING (Deep SORT)                │
│                                                             │
│  • Runs every frame (even on skipped detection frames)      │
│  • Kalman Filter → predicts next position of each track     │
│  • Hungarian Algorithm → matches detections to tracks       │
│  • MobileNet Embedder → appearance feature extraction       │
│  • Assigns persistent Track IDs                             │
└───────────────────────────┬─────────────────────────────────┘
                            │  confirmed tracks with IDs
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RENDERING & OUTPUT                       │
│                                                             │
│  • Draw bounding boxes (per-class color)                    │
│  • Draw label: "classname #ID"                              │
│  • Draw HUD: FPS + active track count                       │
│  • Display window  →  optional save to .mp4                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Component | Library | Version | Purpose |
|---|---|---|---|
| Detection Model | `ultralytics` (YOLOv8) | 8.2.0 | Object detection |
| Tracking Algorithm | `deep-sort-realtime` | 1.3.2 | Multi-object tracking |
| Video Processing | `opencv-python` | ≥ 4.9.0 | Frame capture, drawing, display |
| Deep Learning Backend | `torch` + `torchvision` | ≥ 2.0 | Model inference |
| Numerical Operations | `numpy` | ≥ 1.24.0 | Array handling |

### Why YOLOv8 over Faster R-CNN?

| Criteria | YOLOv8 | Faster R-CNN |
|---|---|---|
| Speed | ⚡ Real-time (~60+ FPS on GPU) | 🐢 ~5–7 FPS |
| Accuracy (mAP) | ✅ Competitive | ✅ Slightly higher |
| Integration ease | ✅ 3-line setup | ❌ Verbose setup |
| COCO pretrained | ✅ Yes | ✅ Yes |
| Best for | Real-time video | Offline batch processing |

### Why Deep SORT over SORT?

SORT uses only bounding box IoU for association — Deep SORT adds an **appearance embedding** (MobileNet) which lets it re-identify objects after occlusion or brief disappearance, resulting in far fewer ID switches in crowded scenes.

---

## 📁 Project Structure

```
object-detection-tracking/
│
├── detect_and_track.py     # Main script — detection + tracking pipeline
├── requirements.txt        # Python dependencies (pinned versions)
├── SETUP_GUIDE.md          # Step-by-step installation guide
├── README.md               # This file
│
├── tracking_env/           # Virtual environment (generated — do not commit)
│
└── outputs/                # Generated files (create manually or auto-generated)
    ├── tracked_output.mp4      # Saved annotated video (--save flag)
    └── screenshot_XXXXX.jpg    # Frame screenshots (press 's')
```

> **Note:** Add `tracking_env/` and `outputs/` to your `.gitignore` if pushing to GitHub.

---

## ⚙️ Installation

### 1. Clone / Download the project

```bash
git clone https://github.com/yourusername/object-detection-tracking.git
cd object-detection-tracking
```

### 2. Create a virtual environment

```bash
python -m venv tracking_env

# Activate — Windows CMD
tracking_env\Scripts\activate

# Activate — Windows PowerShell
tracking_env\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA *(GPU users — do this first)*

Check your CUDA version:
```bash
nvidia-smi
```

Then install the matching PyTorch build:
```bash
# CUDA 12.1 (recommended for RTX 40-series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch torchvision
```

Verify GPU is detected:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> YOLOv8 model weights (`yolov8n.pt`) are **auto-downloaded** on first run — no manual setup needed.

---

## 🚀 Usage

### Run on webcam (default)

```bash
python detect_and_track.py
```

### Run on a video file

```bash
python detect_and_track.py --source path\to\video.mp4
```

### Run and save annotated output

```bash
python detect_and_track.py --source path\to\video.mp4 --save
```

### Controls during runtime

| Key | Action |
|---|---|
| `q` | Quit and close the window |
| `s` | Save screenshot of current frame |

---

## 🔧 Configuration

All tunable parameters are defined at the top of `detect_and_track.py`:

```python
YOLO_MODEL      = "yolov8n.pt"  # Model size: n / s / m / l / x
CONF_THRESHOLD  = 0.45          # Min detection confidence (0.0 – 1.0)
IOU_THRESHOLD   = 0.45          # NMS overlap threshold
MAX_AGE         = 30            # Frames to keep a lost track alive
N_INIT          = 3             # Frames before a new track is confirmed
NN_BUDGET       = 100           # Max appearance features stored per track
FRAME_SKIP      = 1             # Run YOLO every N frames (1 = every frame)
DISPLAY_FPS     = True          # Show FPS overlay on screen
```

### Model size vs. performance trade-off

| Model | Size | Speed (RTX 4050) | mAP50 |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ~80–100 FPS | 37.3 |
| `yolov8s.pt` | 22 MB | ~60–80 FPS | 44.9 |
| `yolov8m.pt` | 52 MB | ~40–55 FPS | 50.2 |
| `yolov8l.pt` | 87 MB | ~25–35 FPS | 52.9 |
| `yolov8x.pt` | 136 MB | ~15–20 FPS | 53.9 |

> For real-time use, `yolov8n` or `yolov8s` is recommended.

### Enable GPU acceleration (RTX 4050)

In `detect_and_track.py`, find the tracker initialization and change:

```python
half=True,           # FP16 half precision — faster inference on RTX
embedder_gpu=True,   # Run MobileNet appearance embedder on GPU
```

---

## 🔬 How It Works

### Detection Phase

Each frame is passed through YOLOv8, which outputs:
- **Bounding boxes** in `[x1, y1, x2, y2]` format
- **Confidence scores** per detection
- **Class IDs** mapped to COCO class names (e.g., `0 → person`)

Detections below `CONF_THRESHOLD` are discarded before being passed to the tracker.

### Tracking Phase (Deep SORT)

Deep SORT maintains a set of **tracks**, each representing a detected object across time. For every new frame:

1. **Kalman Filter** predicts the new position of each existing track
2. **MobileNet embedder** extracts a 128-dim appearance feature vector from each detection
3. **Hungarian Algorithm** solves the optimal assignment between predicted track positions and new detections — using a combination of IoU distance and appearance (cosine) distance
4. Unmatched detections start new **tentative tracks**; tracks confirmed after `N_INIT` frames get a permanent ID
5. Tracks unmatched for more than `MAX_AGE` frames are deleted

### ID Assignment

- IDs are assigned sequentially starting from `1`
- Once an ID is deleted (after `MAX_AGE` frames of absence), it is **not reused**
- Re-entering objects get a new ID unless appearance similarity brings them back within `MAX_AGE`

---

## 📈 Performance

Tested on **Windows 11, RTX 4050 6GB, 16GB RAM**:

| Config | FPS | Notes |
|---|---|---|
| YOLOv8n, CPU only | ~8–12 | Usable for non-real-time |
| YOLOv8n, GPU (FP32) | ~65–80 | Real-time, good accuracy |
| YOLOv8n, GPU (FP16) | ~85–100 | Best speed, minimal accuracy loss |
| YOLOv8s, GPU (FP16) | ~60–75 | Recommended balance |
| YOLOv8m, GPU (FP16) | ~40–55 | Higher accuracy, still smooth |

> FPS values are approximate and depend on scene complexity and number of tracked objects.

---

## 🐛 Troubleshooting

**Webcam not opening**
```bash
# Try different device indices
python detect_and_track.py --source 1
python detect_and_track.py --source 2
```

**`ModuleNotFoundError`**
```bash
# Virtual environment not activated
tracking_env\Scripts\activate
pip install -r requirements.txt
```

**`torch.cuda.is_available()` returns False**
```bash
# CUDA/PyTorch version mismatch — reinstall torch
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Very low FPS on GPU**
- Ensure `embedder_gpu=True` and `half=True` in the script
- Switch to `yolov8n.pt`
- Set `FRAME_SKIP = 2` to detect every other frame

**ID switches / wrong IDs in crowded scenes**
- Lower `CONF_THRESHOLD` to `0.35` to catch more detections
- Increase `MAX_AGE` to `50` to retain lost tracks longer
- Try `yolov8m.pt` for better detection quality

**`deep_sort_realtime` version error**
```bash
pip install deep-sort-realtime==1.3.2 --force-reinstall
```

---

## 🔮 Future Improvements

- [ ] GUI panel to adjust confidence threshold and model at runtime
- [ ] Multi-threading: detection and tracking in separate threads for higher FPS
- [ ] Trajectory logging — export each object's path to CSV
- [ ] Zone-based counting — count objects crossing a defined line or region
- [ ] Custom model support — plug in a fine-tuned YOLO model for specific domains
- [ ] REST API / MQTT stream for integration with dashboards or IoT systems
- [ ] Docker containerization for portable deployment

---

## 📄 License

This project is built for academic and educational purposes using open-source libraries:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) — MIT
- [OpenCV](https://opencv.org/) — Apache 2.0

---

<div align="center">
Built as part of Task 4 — Object Detection and Tracking
</div>
