# Trackmania HUD OCR & AI Monitoring

This project captures and interprets real-time HUD data from **Trackmania Modded Forever** using OCR (Optical Character Recognition).  
It extracts vehicle telemetry (position, speed, orientation, checkpoints, etc.) and feeds it into a monitoring / reward system suitable for AI experimentation or analytics.

---

## Overview

The system works in four main stages:

1. **Window Capture**
2. **HUD Cropping & Image Preprocessing**
3. **OCR Text Extraction**
4. **HUD Parsing & Game Logic Processing**

---

## Features

- 📸 Live screenshot capture of the Trackmania window
- 🧠 OCR-based extraction of HUD telemetry
- 🎯 Semantic parsing using HUD labels (Position, Speed, Checkpoints, etc.)
- ⏱ Race timing and best-time tracking
- 🏁 Checkpoint and finish-line detection
- 📈 Reward calculation (for AI / RL use cases)

---

## Dependencies

- `pygetwindow` – Locate and track the game window
- `mss` – Fast screen capture
- `pytesseract` – OCR engine
- `Pillow` – Image processing
- `opencv-python` – Image preprocessing
- `numpy` – Numerical operations

Tesseract OCR must be installed separately and available in your system PATH.

---

## How It Works

### 1. Window Detection & Screenshot Capture

The game window is located using its title:

```python
gw.getWindowsWithTitle(window_title)
