# 🌊 Aqua Vision

**Real-Time Underwater Marine Species Detection and Enhancement Platform**
using YOLO11 and CLAHE

> Minor Project (ARP-455) · B.Tech 7th Semester
> University School of Automation & Robotics (USAR), Guru Gobind Singh Indraprastha University

## 🔗 Live Demo

**👉 https://aquavision-gaffcuat4dguq8iyggjlla.streamlit.app/**

Upload an underwater image, and the app enhances it, detects marine species and logs the result.

---

## 📌 Overview

Underwater images suffer from colour cast, low contrast and blur because water absorbs red light first and scatters the rest. Standard detectors perform poorly on such frames.

Aqua Vision combines two stages:

1. **Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalisation) applied to the L channel of the LAB colour space.
2. **Detection:** a fine-tuned **YOLO11n** model that detects 7 marine classes.

Everything is wrapped in a **Streamlit** dashboard with an **SQLite** detection history and CSV export.

**Detected classes:** `fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray`

## ✨ Features

- Upload underwater images (JPG / PNG)
- CLAHE enhancement with adjustable clip limit (toggle on/off)
- Adjustable confidence and IoU (NMS) thresholds
- Side-by-side original vs. enhanced + annotated view
- Metric cards (objects detected, highest confidence, distinct species) and class-count chart
- Detection history stored in SQLite, with CSV export

## 🏗️ Architecture

```
Image ─► Pre-processing (RGB→LAB, CLAHE on L, LAB→RGB)
      ─► YOLO11n inference (best.pt) + NMS
      ─► Visualisation & analytics (Streamlit)
      ─► Persistence (SQLite: detection_logs → CSV export)
```

## 📊 Results (validation set: 127 images, 909 instances)

| Metric | Value |
|---|---|
| Precision | 0.818 |
| Recall | 0.693 |
| mAP@0.5 | 0.764 |
| mAP@0.5:0.95 | 0.464 |
| F1-score (from P and R) | 0.750 |
| Best epoch | 37 of 40 |

| Class | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| fish | 0.850 | 0.692 | 0.801 | 0.452 |
| jellyfish | 0.868 | 0.888 | 0.926 | 0.535 |
| penguin | 0.712 | 0.683 | 0.687 | 0.321 |
| puffin | 0.652 | 0.482 | 0.565 | 0.257 |
| shark | 0.866 | 0.649 | 0.734 | 0.488 |
| starfish | 0.878 | 0.667 | 0.760 | 0.584 |
| stingray | 0.897 | 0.788 | 0.872 | 0.610 |

**Latency per frame:** CLAHE 3.2 ms + inference 36.4 ms + NMS 2.6 ms = **42.2 ms (~23.7 FPS end to end)**.

**Model:** YOLO11n, 2,583,517 parameters, 6.4 GFLOPs.

## 🧠 Training Configuration

| Setting | Value |
|---|---|
| Hardware | Google Colab, NVIDIA Tesla T4 (16 GB) |
| Optimiser | AdamW |
| Learning rate | 0.000909 |
| Weight decay | 0.0005 |
| Momentum | 0.9 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Epochs | 40 |
| Data | 447 train / 127 validation images, 7 classes |

## 📁 Project Structure

```
aqua_vision/
├── app.py            # Streamlit dashboard
├── processing.py     # CLAHE enhancement (LAB colour space)
├── detector.py       # YOLO11n inference wrapper
├── database.py       # SQLite logging, history, CSV export
├── requirements.txt
└── models/
    └── best.pt       # fine-tuned YOLO11n weights
```

## 🚀 Run Locally

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd aqua_vision

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure the trained weights are at models/best.pt

# 5. Launch the app
streamlit run app.py
```

Then open http://localhost:8501.

### requirements.txt

```
streamlit>=1.30.0
ultralytics>=8.3.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
```

## 🗄️ Database Schema

Table `detection_logs` in `detections.db`:

| Column | Type | Notes |
|---|---|---|
| id | INTEGER | Primary key, autoincrement |
| timestamp | DATETIME | Default current time |
| filename | TEXT | Uploaded file name |
| total_objects | INTEGER | Number of detections |
| highest_confidence | REAL | Maximum confidence score |
| class_breakdown | TEXT | JSON, e.g. `{"fish": 5, "shark": 1}` |

## ⚠️ Limitations

- Results are from a single validation split; no separate test set yet.
- The effect of CLAHE on detection accuracy (with vs. without enhancement) is still to be measured on real data.
- Puffin and penguin are the weakest classes (small, visually similar).
- CLAHE improves luminance contrast but does not correct colour cast and can amplify noise.
- Streamlit Cloud free tier runs on CPU, so speed may be lower than the figures above.

## 🛣️ Roadmap

- [ ] End-to-end integration testing
- [ ] Ablation study: detection with vs. without CLAHE
- [ ] Video / live-stream processing
- [ ] Colour-correction step and denoising
- [ ] Confusion-matrix analysis and improvement for puffin/penguin
- [ ] Edge deployment (NVIDIA Jetson)

## 🛠️ Tech Stack

Python · PyTorch · Ultralytics YOLO11 · OpenCV · Streamlit · SQLite · Pandas · NumPy

## 👥 Authors

- [Student Name: Tanush Garg]
- Supervisor: Dr. Amrit Pal Singh

## 📚 Key References

- Pizer et al. (1987), Adaptive histogram equalization and its variations
- Zuiderveld (1994), Contrast limited adaptive histogram equalization
- Islam et al. (2020), FUnIE-GAN, IEEE RA-L
- Li et al. (2020), UIEB benchmark, IEEE TIP
- Redmon et al. (2016), YOLO, CVPR
- Khanam & Hussain (2024), YOLOv11 overview, arXiv:2410.17725

Full reference list is in the project report.

## 📄 License

Developed for academic purposes at USAR, GGSIPU. 
