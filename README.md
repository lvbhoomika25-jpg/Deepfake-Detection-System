# Deepfake Detection System

A desktop-based deepfake detection prototype for analyzing images and videos.

## Overview

This project provides a graphical interface for selecting image or video files and analyzing them for possible manipulation. It performs face detection and extracts several image characteristics, including frequency, texture, color, edge, and compression-related features.

## Features

- Image and video input
- OpenCV-based face detection
- Feature-based deepfake analysis
- Confidence score
- REAL / FAKE / UNCERTAIN / NO FACE results
- Image preview and video information
- Detailed feature analysis
- Feature visualization charts
- Batch processing of multiple media files
- Adjustable confidence threshold
- Adjustable analysis depth
- Dark-themed desktop GUI

## Technologies Used

- Python
- CustomTkinter
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

## Project Structure

```text
Deepfake-Detection-System/
├── advanced_deepfake_gui.py
├── requirements.txt
└── README.md
```

## How to Run

1. Install Python 3.10+.
2. Open a terminal in this project folder.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python advanced_deepfake_gui.py
```

## Note

The current implementation uses feature-based analysis and a rule-based probability calculation when a trained classifier is not available. It should be treated as a project prototype and not as a forensic-grade deepfake detector.
