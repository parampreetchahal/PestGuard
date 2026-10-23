# PestGuard

**Crop pest classification and detection with a multilingual advisory web app.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-purple)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![Supabase](https://img.shields.io/badge/Supabase-Storage-green?logo=supabase)
![Redis](https://img.shields.io/badge/Redis-Caching-red?logo=redis)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Live demo:** [ADD LINK HERE]  |  **Demo video:** [ADD LINK HERE]

![PestGuard screenshot](docs/screenshot-home.png)

## What it does

A farmer uploads a photo of a crop pest. PestGuard identifies the pest (12 classes), returns treatment and prevention advice in the user's language, and shows nearby pesticide suppliers on Google Maps.

## Results

| Model | Task | Test accuracy | Precision | Recall | mAP@0.5 |
|---|---|---|---|---|---|
| PestNet (CNN from scratch) | Classification | [FILL] | [FILL] | [FILL] | n/a |
| HybridPestNet (EfficientNet-B3 + custom CNN) | Classification | 92.10% | 0.93 | 0.92 | n/a |
| YOLOv8 | Detection | n/a | [FILL] | [FILL] | 0.943 |

Dataset: 8,400+ images, 12 pest classes, [ADD SOURCE, e.g. Kaggle dataset name], split [ADD RATIO] into train / validation / test, with augmentation applied to the training set only.

Add the confusion matrix for HybridPestNet here: `docs/confusion-matrix.png`

## Architecture

```
Image upload -> preprocessing -> HybridPestNet classification
                                   |-> pest details + treatment advice ([ADD API/MODEL NAME])
                                   |-> Google Maps API -> nearby pesticide suppliers
Supabase: data storage    Redis: response caching
```

## Tech stack

- **ML / CV:** TensorFlow, Keras, EfficientNet-B3, YOLOv8 (Ultralytics), OpenCV, NumPy, Pandas
- **Backend:** Flask, REST APIs, Supabase, Redis
- **APIs:** Google Maps API, [ADD RECOMMENDATION API/MODEL NAME]
- **Deployment:** Procfile-based deployment on [ADD PLATFORM]

## Repository structure

```
PestGuard/
├── app.py                                  # Flask application
├── test.py                                 # inference test script
├── best_pestnet.keras                      # trained model weights
├── class_labels.json                       # class index -> label
├── class_names.json                        # [REMOVE if duplicate of class_labels.json]
├── hybrid-approach.ipynb                   # HybridPestNet training
├── custom-cnn-from-scratch.ipynb           # PestNet training
├── YOLOv8_HybridPest_Pipeline_Kaggle.ipynb # YOLOv8 training
├── requirements.txt
├── runtime.txt
├── Procfile
├── static/
└── templates/
```

## Run locally

```bash
git clone https://github.com/parampreetchahal/PestGuard.git
cd PestGuard
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with these variables (never commit it):

```
SUPABASE_URL=
SUPABASE_KEY=
REDIS_URL=
GOOGLE_MAPS_API_KEY=
[ADD ANY OTHER KEYS]
```

```bash
python app.py
```

## Limitations

- Trained on a single public dataset of 12 pest classes; accuracy on field photos with different lighting or backgrounds is untested.
- Treatment advice is AI-generated and should be verified with a local agricultural officer before use.
- [ADD ANY OTHER REAL LIMITATION, e.g. inference time, unsupported languages]

## Future work

- Disease detection alongside pests
- Mobile app and offline inference (TFLite / ONNX export)
- Weather-aware recommendations

## Author

**Parampreet Singh** | B.Tech CSE, Graphic Era Hill University
[GitHub](https://github.com/parampreetchahal) | [LinkedIn](https://www.linkedin.com/in/parampreet-singh23/) | [Portfolio](https://parampreetsingh.netlify.app)

## License

MIT. See [LICENSE](LICENSE).
