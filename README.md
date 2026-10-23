<div align="center">

# PestGuard
### AI-Powered Pest Detection & Smart Agricultural Advisory System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://tensorflow.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)](https://flask.palletsprojects.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent AI-powered platform that detects crop pests from images, provides multilingual recommendations, and helps farmers identify nearby pesticide suppliers using location-aware services.

</div>

---

# Overview

PestGuard is an end-to-end Artificial Intelligence and Computer Vision based agricultural advisory platform designed to assist farmers in early pest identification and treatment.

The system combines Deep Learning, Object Detection, Computer Vision, Natural Language Processing, and Geolocation Services to provide an intelligent decision support system capable of:

- Detecting crop pests from uploaded images
- Classifying pests using multiple AI models
- Generating multilingual treatment recommendations
- Providing pesticide information
- Locating nearby pesticide suppliers
- Delivering an easy-to-use web interface for farmers

The project was developed as a research-oriented academic project focusing on practical AI applications in precision agriculture.

---

# Key Features

## AI Pest Detection

- HybridPestNet (EfficientNet-B3 + Custom CNN)
- PestNet (CNN trained from scratch)
- YOLOv8 Real-Time Pest Detection
- Automatic image preprocessing
- High accuracy classification

---

## Smart Recommendations

- AI-generated pest information
- Treatment recommendations
- Prevention techniques
- Crop protection suggestions
- Multilingual responses

---

## Location Services

- Google Maps Integration
- Nearby pesticide suppliers
- Location-aware recommendations
- Navigation support

---

## User Friendly Interface

- Modern responsive UI
- Image upload support
- Fast prediction results
- Simple dashboard
- Mobile-friendly design

---

# System Architecture

```
                User Uploads Image
                        │
                        ▼
             Image Preprocessing
                        │
                        ▼
      ┌─────────────────────────────────┐
      │                                 │
      │   HybridPestNet Classification  │
      │                                 │
      └─────────────────────────────────┘
                        │
                        ▼
             Pest Identification
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
 AI Recommendations   Pest Details   Treatment
        │
        ▼
 Google Maps API
        │
        ▼
 Nearby Pesticide Shops
```

---

# Technology Stack

## Programming Languages

- Python
- HTML
- CSS
- JavaScript

## Machine Learning

- TensorFlow
- Keras
- YOLOv8
- EfficientNet-B3
- OpenCV
- NumPy
- Pandas

## Backend

- Flask
- REST APIs

## Database

- PostgreSQL

## APIs

- Google Maps API
- AI Recommendation API

## Development Tools

- VS Code
- Git
- GitHub
- Jupyter Notebook

---

# Deep Learning Models

## 1. HybridPestNet

A hybrid architecture combining EfficientNet-B3 feature extraction with a custom CNN classifier for improved pest recognition.

### Advantages

- Higher accuracy
- Better feature extraction
- Faster convergence
- Robust predictions

---

## 2. PestNet

A lightweight CNN model trained from scratch specifically for pest image classification.

### Advantages

- Lightweight
- Faster inference
- Low computational cost

---

## 3. YOLOv8

Real-time object detection model used for detecting pest regions within images.

### Advantages

- Real-time inference
- Multiple object detection
- High precision
- Bounding box localization

---

# Dataset

- Approximately **8,400+ images**
- **12 Pest Classes**
- Data Augmentation
- Image Normalization
- Training, Validation, and Testing Split

---

# Performance

| Model | Performance |
|--------|------------|
| HybridPestNet | **92.10% Test Accuracy** |
| YOLOv8 | **mAP@0.5 = 0.943** |

---

# Project Structure

```
PestGuard/

│
├── app.py
├── requirements.txt
├── config.py
├── runtime.txt
├── Procfile
│
├── static/
│
├── templates/
│
├── hybrid-approach.ipynb
├── custom-cnn-from-scratch.ipynb
├── YOLOv8_HybridPest_Pipeline_Kaggle.ipynb
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/parampreetchahal/PestGuard.git

cd PestGuard
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create your configuration file (or use environment variables) with the required API keys and database credentials.

> **Note:** Sensitive credentials are intentionally excluded from this repository.

---

## Run Application

```bash
python app.py
```

# Future Enhancements

- Mobile Application
- Voice-based Assistant
- Disease Detection
- Weather Integration
- Fertilizer Recommendation
- IoT Sensor Support
- Drone-based Monitoring
- Cloud Deployment

---

# Research Contribution

This project demonstrates the practical application of Artificial Intelligence, Deep Learning, Computer Vision, and Agricultural Technology to improve early pest detection and assist farmers in making informed crop protection decisions.

---

# Author

### Parampreet Singh

B.Tech Computer Science Engineering

Graphic Era Hill University, Dehradun

GitHub:
https://github.com/parampreetchahal

LinkedIn:
https://www.linkedin.com/in/parampreet-singh23/

---

# License

This project is intended for educational, research, and demonstration purposes.
