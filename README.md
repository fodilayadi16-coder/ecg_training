# Project Title

A comprehensive deep learning solution for classifying ECG beats, featuring real-time data acquisition from ESP32 with AD8232 sensor and an interactive web visualization dashboard.
---

## Overview

This project aims to:
- Develop a deep learning model which can classify different beats of ECG into different classes
- Test the model using ESP32 data and ECG sensor AD8232 for real-time classification
- Model inference via Raspberry Pi
- Create a web application to visualize the ECG signal and different feautures (classification, heart rate, etc)

---

## Features

- Feature 1 (e.g., Data preprocessing and model deployement)
- Feature 2 (e.g., Real-time data acquisition from ESP32)
- Feature 3 (e.g., Signal preprocessing and noise filtering)
- Feature 4 (e.g., CNN-based classification)
- Feature 5 (e.g., Visualization with web app)

---

## Project Structure

```bash
ecg_training/
├── data/                    # Data directory
│   ├── raw/                 # Raw ECG datasets (MIT-BIH, etc.)
│   └── processed/           # Cleaned and transformed data
├── src/                     # Source code
│   ├── preprocessing/       # Signal filtering, segmentation
│   ├── models/              # CNN architectures
│   ├── training/            # Model training pipelines
│   └── evaluation/          # Performance metrics & visualization
│   ├── experiments/         # Cross dataset test
│   ├── loaders/             # Loading data
│   └── utils/               # Functions
├── requirements.txt         # Python package dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

