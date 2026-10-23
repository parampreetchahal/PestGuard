<div align="center">

# PestGuard

### AI-powered pest detection and agricultural advisory

**Identify crop pests. Understand the result. Explore practical next steps.**

[▶ Watch the live project demo](https://drive.google.com/file/d/1dP3HAEQdJGcgYdoDYb72JZjAOI4AnfqC/view?usp=drive_link) · [Browse the source code](https://github.com/parampreetchahal/PestGuard)

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-7B61FF)](https://docs.ultralytics.com/)
[![Supabase](https://img.shields.io/badge/Storage-Supabase-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com/)

</div>

---

## At a glance

PestGuard is a computer-vision web application built to help users identify crop pests from images and access pest information, treatment and prevention guidance, and location-based supplier discovery. It combines deep-learning experiments with a Flask application and supporting cloud services.

| Reported result | Value |
|---|---:|
| HybridPestNet test accuracy | **92.10%** |
| Standalone PestNet CNN test accuracy | **76.42%** |
| YOLO-assisted pipeline accuracy | **88.17%** |
| YOLO detector mAP@0.5 | **0.943** |
| Classification dataset | **8,400 images · 12 classes** |

<sub>Metrics are reproduced from the project report, Chapter 5. Classification accuracy and YOLO localization metrics describe different evaluation tasks and are not directly interchangeable.</sub>

## Live demo

Watch the shared live-demo recording: [Open PestGuard demo video](https://drive.google.com/file/d/1dP3HAEQdJGcgYdoDYb72JZjAOI4AnfqC/view?usp=drive_link). If Google Drive requests access, sign in with an account that has permission to view the file.

## Why PestGuard?

- **Image-based pest identification** using a trained classification model.
- **Multiple model approaches:** HybridPestNet, a standalone CNN (PestNet), and a YOLOv8-assisted detection pipeline.
- **Pest guidance** covering pest details, treatment suggestions, and prevention information.
- **Multilingual advisory experience** intended to make information more accessible.
- **Location-aware discovery** for nearby pesticide suppliers.
- **Web application** built with Flask, with supporting database, caching, email, and request-control services.

## Results and model evaluation

| Model / pipeline | Reported metric | Reported result |
|---|---|---:|
| HybridPestNet (EfficientNet + custom CNN head) | Test accuracy | **92.10%** |
| Standalone PestNet CNN | Test accuracy | **76.42%** |
| YOLO-assisted pipeline | Pipeline accuracy | **88.17%** |
| YOLOv8 detector | mAP@0.5 | **0.943** |

### HybridPestNet per-class F1 score

![HybridPestNet per-class F1 score chart](docs/assets/hybridpestnet-f1.svg)

### HybridPestNet confusion matrix

![HybridPestNet confusion matrix transcribed from the project report](docs/assets/hybridpestnet-confusion-matrix.svg)

The confusion matrix is transcribed from Figure 5.3 in the project report. It is included to make the reported evaluation easier to inspect; it is not a newly generated evaluation from model predictions.

## System architecture

![PestGuard system architecture](docs/assets/architecture.svg)

**High-level flow:** image upload → preprocessing → model inference → pest result → advisory and location-aware services → results presented to the user.

The application code also integrates Supabase and Upstash Redis. Advisory and geolocation features depend on the configuration and external services enabled for a deployment.

## Technology stack

| Layer | Technologies |
|---|---|
| Application | Python, Flask, HTML, CSS, JavaScript |
| Classification | TensorFlow / Keras, EfficientNet-based hybrid model, custom CNN |
| Object detection | YOLOv8 / Ultralytics |
| Image processing | Pillow, NumPy and preprocessing utilities |
| Data and caching | Supabase, Upstash Redis |
| Supporting services | Email, geocoding / location lookup, rate limiting and compression |

## Pest classes

The report evaluates these 12 classification labels:

**Aphids · Beetle · Bollworm · Bugs · Caterpillar · Earwig · Grasshopper · Mites · Slug · Snail · Weevil · Whitefly**

## Repository layout

```text
PestGuard/
├── app.py
├── config.py                 # local-only credentials; never commit
├── config.example.py         # redacted configuration template
├── best_pestnet.keras        # trained model expected by the app
├── class_names.json
├── requirements.txt          # original environment export
├── requirements-app.txt      # app runtime dependency set
├── templates/
├── static/
├── hybrid-approach.ipynb
├── custom-cnn-from-scratch.ipynb
├── YOLOv8_HybridPest_Pipeline_Kaggle.ipynb
└── docs/
    └── assets/
        ├── architecture.svg
        ├── hybridpestnet-f1.svg
        └── hybridpestnet-confusion-matrix.svg
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/parampreetchahal/PestGuard.git
cd PestGuard
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-app.txt
```

> **Why `requirements-app.txt`?** The original `requirements.txt` was exported from a larger local environment and contains machine-specific package paths. `requirements-app.txt` lists the runtime dependencies used by the Flask app. It is a starting point for a clean install, not a lockfile; verify the environment before deployment.

### 4. Configure credentials

Copy the redacted template to a local configuration file and replace each placeholder with your own service credentials:

```bash
# macOS / Linux
cp config.example.py config.py
```

On Windows PowerShell:

```powershell
Copy-Item config.example.py config.py
```

The app references Supabase, Upstash Redis, Flask session, email, and Groq API settings. Configure only the services you intend to use. The local `config.py` is ignored by Git.

**Never commit API keys, passwords, tokens, or production credentials.**

### 5. Start Flask

```bash
python app.py
```

The application loads `best_pestnet.keras` and `class_names.json` at startup. Ensure both files are present at the expected paths. Some features may also require correctly configured external services.

## Reproducibility and interpretation

- The classification dataset is reported as 8,400 images across 12 classes, split into training, validation, and test sets in an 80:10:10 ratio.
- HybridPestNet uses 300 × 300 input images in the report; the standalone CNN uses 224 × 224.
- The YOLO detection dataset is reported separately as 4,300 manually annotated images.
- These are project-report results, not a guarantee of performance on new crops, camera conditions, or geographic regions.
- Pest identification and treatment suggestions should be treated as decision support, not a substitute for qualified agricultural advice or pesticide-label instructions.

## Project status

PestGuard is an academic/research prototype. Deployment readiness depends on environment configuration, portable dependency management, model availability, and the external services used by the application.

## Contributors

Developed as a team project at **Graphic Era Hill University, Dehradun**.

- **Parampreet Singh** — [GitHub](https://github.com/parampreetchahal) · [LinkedIn](https://www.linkedin.com/in/parampreet-singh23/)
- Project teammates and mentor are acknowledged in the academic project report.

## License

No explicit license file was verified during this documentation pass. Add a repository license before describing the project as open source or specifying reuse permissions.
