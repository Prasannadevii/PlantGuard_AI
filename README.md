# 🌱 PlantGuard AI

**AI-powered plant health decision support** - disease classification, explainable AI, lesion segmentation,
severity estimation, environmental risk, disease-progression projection, recommendations, plant history
monitoring and a Groq-powered assistant, in one dashboard.

> Upload (or scan) a leaf → *What plant? What disease? How sure? Where on the leaf? How much is affected?
> What conditions raise the risk? How might it progress? What should I do? How do I track it over time?*

PlantGuard AI is a **decision-support system, not a laboratory diagnosis.** Every important output is shown with
its evidence (probability chart, Grad-CAM, segmented leaf, factor contributions, forecast chart), and every
module is labelled honestly as *trained model*, *heuristic* or *rule-based projection*.

---

## Table of contents
1. [Features](#1-features)
2. [How it works](#2-how-it-works)
3. [Requirements](#3-requirements)
4. [Quick start](#4-quick-start)
5. [Using the dashboard](#5-using-the-dashboard)
6. [Configuration](#6-configuration)
7. [What is real ML and what is not](#7-what-is-real-ml-and-what-is-not)
8. [Model training and evaluation](#8-model-training-and-evaluation)
9. [API reference](#9-api-reference)
10. [Project structure](#10-project-structure)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations](#13-limitations)
14. [Security](#14-security)
15. [Roadmap](#15-roadmap)
16. [Dataset and credits](#16-dataset-and-credits)

---

## 1. Features

| # | Feature | Where |
|---|---|---|
| 1 | Two inputs: **📁 Upload Leaf Image** (JPG/PNG/WebP) and **📷 Scan Leaf** (live camera with guide box and live hints) | dashboard |
| 2 | **Image quality gate** before prediction: leaf present, single leaf, sharpness, lighting, quality score 0-100 | `app/vision.py` |
| 3 | Automatic **leaf crop** so off-centre phone photos resemble the training images | `app/vision.py` |
| 4 | **Crop + disease classification** (15 classes: pepper, potato, tomato) with top-3 probability chart | notebook + `app/pipeline.py` |
| 5 | **Calibrated confidence** and refusal to diagnose when unsure | `calibration.json` |
| 6 | **Grad-CAM** heatmap and overlay (explains the model, not biology) | `app/pipeline.py` |
| 7 | **Lesion segmentation** (mask + overlay: green = healthy, red = affected) | `app/vision.py` |
| 8 | **Severity %** = lesion pixels ÷ leaf pixels × 100, with level (Low/Moderate/High/Critical) | `app/vision.py` |
| 9 | **Environmental risk score** (0-100) with per-factor contribution bars | `app/risk.py` |
| 10 | **Progression projection** for days 0-14 with scenario spread band | `app/risk.py` |
| 11 | **Recommendations** with priority and a "why this recommendation?" evidence table | `app/recommend.py` |
| 12 | **Plant history**: save scans, timeline chart, **predicted-vs-actual** with MAE/RMSE/R² | `app/db.py`, `/api/plants/{id}/history` |
| 13 | **PlantGuard AI Assistant** (Groq) that explains the structured results | `app/chat.py` |
| 14 | **Model evaluation page** read from the real `metrics.json` | dashboard |

Failure cases never produce a fake diagnosis: *no leaf*, *multiple leaves*, *low quality*, *low confidence* and
*unreadable image* each return a message plus guidance.

---

## 2. How it works

```
 Upload / Camera scan
        │
        ▼
 Image quality gate ──fail──► message + tips (no diagnosis)
        │ ok
        ▼
 Leaf crop (square, 384×384)
        │
        ▼
 CNN classifier ──low confidence──► "capture a clearer image" (no diagnosis)
   (224×224, calibrated)
        │
        ├──► Grad-CAM heatmap + overlay
        │
        ├──► Lesion segmentation ──► Severity % + level
        │
        ├──► Risk (rule-based) ◄── temperature, humidity, rainfall, recent rain, severity
        │         │
        │         ▼
        │    Progression projection (days 0-14, median + spread)
        │         │
        └─────────┴──► Recommendation engine ──► JSON report ──► Dashboard
                                                       │
                                                       ├──► SQLite history
                                                       └──► Groq assistant (explains the JSON only)
```

Stack: **Python, TensorFlow/Keras, OpenCV, NumPy, FastAPI, SQLite, Groq API, vanilla JS + Chart.js.**

---

## 3. Requirements

- Python 3.10-3.12 recommended (check the TensorFlow install page for the versions supported by your TensorFlow release)
- TensorFlow ≥ 2.16 (Keras 3) - **use the same version for training and serving**
- ~2 GB free disk for the dataset, a GPU for training (Kaggle's free GPU is enough)
- A modern browser; camera scanning needs `http://localhost` or HTTPS
- Optional: a Groq API key for the chatbot (<https://console.groq.com>)

---

## 4. Quick start

### Step 1 - Train and save the model

1. Get the dataset: <https://www.kaggle.com/datasets/emmarex/plantdisease>
2. Open `notebooks/plantguard_training.ipynb`.

**On Kaggle (easiest)**
- New notebook → *Import* the `.ipynb` → *Add data* → search `plantdisease` (emmarex).
- Settings → enable **GPU** and **Internet**.
- Run once with `QUICK_TEST = True` (a few minutes) to make sure everything works, then set `QUICK_TEST = False` and *Run All*.
- When it finishes, download `plantguard_model_files.zip` from the *Output* tab and **unzip it into `plantguard-ai/model_weights/`**.

**Locally**
- Extract the dataset into `plantguard-ai/datasets/` (or set the environment variable `DATA_ROOT`).
- Start Jupyter **from the `notebooks/` folder** and *Run All*. Files are written to `model_weights/` automatically.

After a real run, `model_weights/` contains:

| File | Purpose |
|---|---|
| `plantguard_model.keras` | the trained model |
| `class_names.json` | class order used by the model |
| `calibration.json` | temperature and low-confidence threshold |
| `metrics.json` | all evaluation numbers shown in the app |
| `confusion_matrix.png`, `training_curves.png` | evaluation figures |
| `train/validation/test_split.csv`, `checkpoints/` | reproducibility |

### Step 2 - Run the app

```bash
cd plantguard-ai
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env   (optional: add GROQ_API_KEY)
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.

If the model files are missing, the page shows a red banner explaining what to do - nothing is faked.

---

## 5. Using the dashboard

1. **Pick or create a plant** (one entry per physical plant - history is tracked per plant).
2. **Add a leaf**: *Upload Leaf Image*, or *Scan Leaf* → place one leaf inside the dashed box, follow the live hint, press *Capture*.
3. *(Optional)* enter **temperature, humidity, rainfall** and tick **recent rain**. Without them the risk uses neutral values and the forecast is flagged as less certain.
4. Set the **scan date** (defaults to today; change it to back-date older photos).
5. Press **Analyze**. Scroll through the report:
   - four summary cards (disease, severity, risk, confidence),
   - *Disease analysis*: top predictions + original / Grad-CAM heatmap / overlay,
   - *Severity analysis*: original / mask / overlay + donut chart + pixel counts,
   - *Environmental risk*: factor bars with weights and contributions,
   - *Disease progression*: line chart with spread band + "why" list,
   - *Recommendations*: priority, actions, evidence table, next-scan interval.
6. **Scan the same plant again later** → *Plant monitoring* shows the timeline and scores the earlier forecast against reality.
7. Click **🤖 Ask PlantGuard** to ask *"What should I do now?"*, *"Why is the risk high?"*, etc.

Tips for good photos: one leaf, plain background, even light, fill ~1/3 of the frame, hold the camera steady.

---

## 6. Configuration

Environment variables (put them in `.env`; never commit that file):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | enables the chatbot; used by the backend only |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id - Groq renames/retires models, check <https://console.groq.com/docs/models> if chat fails |
| `PG_LOW_CONF` | `0.60` | confidence below which no diagnosis is shown (overridden by `calibration.json` when present) |
| `PG_WEIGHTS_DIR` | `model_weights/` | folder with the trained model files |
| `DATA_ROOT` | - | *(notebook only)* dataset location when running locally |

Tunable constants live in code: image-quality thresholds and severity bands in `app/vision.py`, rule-model
parameters in `app/knowledge.py`. See the Developer Guide.

---

## 7. What is real ML and what is not

| Module | Method | Trained on data? |
|---|---|---|
| Image quality gate | OpenCV sharpness / brightness / leaf-area / single-leaf checks | No (thresholds - tune on your photos) |
| Crop + disease | EfficientNetB0 or MobileNetV2 (best on validation macro-F1), temperature-calibrated | **Yes - PlantVillage** |
| Low-confidence rejection | Calibrated max-probability threshold | Threshold informed by validation coverage table |
| Grad-CAM | Gradients of the predicted class w.r.t. the last feature map | n/a (explains the model) |
| Segmentation + severity | **Colour-based HSV heuristic** | **No** - unvalidated until you measure IoU/Dice |
| Environmental risk | **Rule-based weighted model** | **No** - expert assumptions needing citations |
| Progression | **Logistic growth projection** + Monte-Carlo spread; growth rate partly fitted to the plant's own history after ≥3 scans | **No** - a projection, not a validated forecast |
| Prediction vs actual | Stored forecast vs the next real scan | Computed from your real scans |
| Recommendations | Rule engine (cultural measures only, no pesticide doses) | No |
| Chatbot | Groq LLM given only the structured JSON | n/a |

There is no public dataset that links PlantVillage-style images with weather and severity over time. Training a
regressor on a simulator I wrote would only recover my own formula, so the risk and progression modules are
deliberately **not** presented as trained ML. Full details: `docs/METHODS.md`.

---

## 8. Model training and evaluation

The notebook (`notebooks/plantguard_training.ipynb`) runs these stages:

1. **Dataset discovery and report** - classes, counts, imbalance, image sizes.
2. **Cleaning** - unreadable files removed; exact duplicates (MD5) dropped; files appearing under two labels dropped.
3. **Leakage-safe 70/15/15 split** - PlantVillage has rotated/flipped copies of the same leaf. Each image gets a 16×16 grayscale fingerprint; images that correlate strongly under any of 8 rotations/flips are grouped, and whole groups go to one split. The correlation threshold is chosen automatically so no group exceeds 40 images. Assertions verify no group and no exact duplicate crosses splits.
4. **Augmentation** - flips, rotation, zoom, brightness, contrast (no hue changes: colour is diagnostic).
5. **Model comparison on validation** - baseline CNN, MobileNetV2, EfficientNetB0; same data, seed, class weights. Best chosen on **validation macro-F1**.
6. **Fine-tuning** of the best model's top 30 backbone layers at lr 1e-5; kept only if validation macro-F1 improves.
7. **One test-set evaluation** - accuracy, macro precision/recall/F1, per-class table, confusion matrix, train/val curves, overfitting gap.
8. **Calibration** - temperature scaling fitted on validation; ECE before/after; coverage-vs-accuracy table at confidence thresholds.
9. **Save + verify** - model reloaded from disk, predictions compared, Grad-CAM layer checked.

**Reporting rule:** the numbers on the dashboard's *Model evaluation* section come from `metrics.json`, written from
the real held-out test set. Do not quote accuracy before you have run the notebook, and never present
`QUICK_TEST` results (the page warns when `metrics.json` came from a quick test).

PlantVillage images are lab-style, so accuracy on phone photos is expected to be lower. Measure it with your own
photos and an external set such as PlantDoc, and report the drop honestly.

---

## 9. API reference

Base URL `http://127.0.0.1:8000`. Full interactive docs at `/docs`.

| Method & path | Purpose |
|---|---|
| `POST /api/analyze` | full pipeline (multipart form: `image`, `plant_id`, `temperature`, `humidity`, `rainfall_mm`, `recent_rain`, `observed_at`) |
| `POST /api/quality-check` | image-quality gate only (`image`) |
| `GET /api/plants` / `POST /api/plants` | list / create plants (`{"name": "...", "crop": "...", "location": "..."}`) |
| `GET /api/plants/{id}/history` | scans + predicted-vs-actual evaluation |
| `GET /api/analyses/{id}` | a stored report |
| `POST /api/chat` | `{"analysis_id": 1, "message": "...", "history": []}` → `{"reply": "..."}` |
| `GET /api/metrics` | contents of `metrics.json` (or `{"available": false}`) |
| `GET /api/metrics/figure/{name}` | `confusion_matrix.png` or `training_curves.png` |
| `GET /api/health` | model status |

**Example**

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F image=@leaf.jpg -F temperature=29 -F humidity=82 -F rainfall_mm=18 -F recent_rain=true
```

**Response (status `ok`, abridged; numbers are illustrative)**

```json
{
  "status": "ok",
  "analysis_id": 7,
  "quality": {"status": "ok", "score": 91, "checks": [ ... ], "tips": []},
  "crop": {"name": "Tomato", "confidence": 97.4},
  "disease": {"name": "Early blight", "class_name": "Tomato_Early_blight", "confidence": 94.2, "is_healthy": false},
  "top3": [{"label": "Tomato - Early blight", "probability": 94.2}, ...],
  "images": {"original": "/files/analyses/7/original.jpg", "heatmap": "...", "cam_overlay": "...", "mask": "...", "seg_overlay": "..."},
  "severity": {"applicable": true, "affected_percent": 23.0, "healthy_percent": 77.0, "level": "Moderate",
               "leaf_pixels": 245000, "diseased_pixels": 56350},
  "environment": {"temperature": 29, "humidity": 82, "rainfall_mm": 18, "recent_rain": true},
  "risk": {"score": 82, "level": "High", "method": "Rule-based weighted model (...)", "factors": [ ... ]},
  "progression": {"milestones": {"day_3": {"median": 31, "low": 29, "high": 34}, "day_7": {...}, "day_14": {...}},
                  "curve": [ ... 15 points ... ], "uncertainty_note": "..."},
  "recommendation": {"priority": "HIGH", "actions": [ ... ], "next_scan_days": "2-3", "evidence": {...}, "disclaimer": "..."},
  "limitations": [ ... ]
}
```

**Non-ok statuses** (`low_quality`, `no_leaf`, `multiple_leaves`, `low_confidence`, `invalid_image`) return
`{"status": ..., "message": ..., "quality": ...}` and contain **no disease field**. They are not saved to the history.

HTTP errors: `400` unsupported type / bad input, `404` unknown plant or analysis, `413` file > 15 MB,
`503` model not loaded.

---

## 10. Project structure

```
plantguard-ai/
├── README.md                  this file
├── DEVELOPER_GUIDE.md         architecture, internals, how to extend
├── requirements.txt
├── .env.example               copy to .env
├── app/
│   ├── main.py                FastAPI routes
│   ├── pipeline.py            ModelService (TensorFlow) + run_analysis orchestrator
│   ├── vision.py              quality gate, leaf crop, segmentation, severity, overlays
│   ├── risk.py                environmental risk, progression projection, forecast evaluation
│   ├── recommend.py           recommendation engine
│   ├── knowledge.py           class metadata, rule parameters, management actions
│   ├── chat.py                Groq assistant
│   ├── db.py                  SQLite storage
│   ├── config.py              paths, env vars
│   └── static/index.html      dashboard (upload, camera, charts, history, evaluation, chat)
├── notebooks/plantguard_training.ipynb
├── model_weights/             created by the notebook
├── data/                      runtime: plantguard.db, analyses/<id>/*.jpg
├── tests/test_logic.py
└── docs/METHODS.md            formulas, parameter tables, assumptions
```

---

## 11. Testing

```bash
python tests/test_logic.py
```

12 tests, no TensorFlow or FastAPI required. They use **synthetic leaves** to verify the mechanics: severity
arithmetic, quality gate (dark/blurry/no-leaf/two-leaf), risk direction, forecast properties (monotone, bounded,
reproducible, uses history), forecast scoring, recommendation priority, database round-trip, and the whole
pipeline with a fake model. They say nothing about real-world model accuracy - that comes from `metrics.json`.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Red banner "Model not loaded" | Run the notebook and put its output files in `model_weights/` |
| Error loading the model / Keras version message | Install the TensorFlow version printed at the top of the notebook run |
| Notebook: "PlantVillage folder not found" | Kaggle: attach the dataset. Local: extract into `datasets/` or set `DATA_ROOT` |
| Notebook: ImageNet weights download fails | Enable Internet on Kaggle, or set `USE_PRETRAINED = False` (much lower accuracy) |
| Notebook warns about no small groups | The fingerprint grouping fell back to per-image groups; inspect the printed thresholds (see Developer Guide) |
| Scan Leaf does nothing | Camera needs `localhost`/HTTPS and permission; use Upload instead |
| Good leaf rejected as "no leaf" / "low quality" | Use a plainer background and better light; tune thresholds in `app/vision.py` |
| Many photos give "low confidence" | Phone photos differ from PlantVillage; add field images and retrain, or lower `PG_LOW_CONF` knowingly |
| Chat says "not configured" | Add `GROQ_API_KEY` to `.env` and restart |
| Chat says unavailable (400/404) | `GROQ_MODEL` may be retired - pick a current id from the Groq console |
| Severity looks wrong | Segmentation is a colour heuristic: needs one leaf on a plain background; validate it (Developer Guide §9) |

---

## 13. Limitations

- **Domain shift** - trained on lab-style PlantVillage images; field accuracy is usually lower.
- **Three crops / 15 classes** - other plants are only caught through low confidence.
- **Segmentation is unvalidated** - hand-label ~50 leaves and report IoU/Dice before relying on severity numbers. Viruses, mites and yellowing are measured as colour change, not discrete lesions.
- **Severity bands** (≤10 / ≤30 / ≤60 / >60 %) are illustrative, not a published standard.
- **Risk and progression parameters are assumptions** - add citations (`docs/METHODS.md`) and adjust `app/knowledge.py`. The forecast band is scenario spread, not a statistical confidence interval.
- **Prediction-vs-actual** is noisy when photos are taken under different conditions and meaningless with 1-2 pairs.
- **Grad-CAM** shows what influenced the model, not biological causality.
- Recommendations are general cultural measures; they never give pesticide products or doses.

---

## 14. Security

- API keys live only in the server's environment / `.env`; the browser never sees them. `.env` is git-ignored.
- Uploads are limited to JPG/PNG/WebP and 15 MB, decoded with Pillow, and never executed.
- All server-provided text is HTML-escaped in the dashboard; chatbot replies are inserted as plain text.
- The app has **no authentication** - run it locally or behind a reverse proxy with auth before exposing it publicly.

---

## 15. Roadmap

Weather-API lookup behind a backend service (e.g. Open-Meteo) · segmentation IoU/Dice report ·
PlantDoc / field-image evaluation · trained U-Net segmenter · user accounts · ONNX export ·
PostgreSQL · Next.js frontend · scheduled re-scan reminders.

---

## 16. Dataset and credits

- **Dataset:** PlantVillage, as distributed on Kaggle: <https://www.kaggle.com/datasets/emmarex/plantdisease>.
  Please cite the original PlantVillage authors (Hughes & Salathé, 2015, *An open access repository of images on plant health to enable the development of mobile disease diagnostics*) and check the dataset's license terms before redistribution.
- **Models:** Keras Applications (EfficientNetB0, MobileNetV2) with ImageNet weights.
- **Libraries:** TensorFlow/Keras, OpenCV, NumPy, Pillow, scikit-learn, SciPy, FastAPI, Uvicorn, Chart.js.
- **LLM:** Groq API (Llama models) for the explanation assistant only.

Built as an academic project. Not for clinical, regulatory or commercial agronomic decisions without expert
verification.
