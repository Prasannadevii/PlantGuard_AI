# 🌱 PlantGuard AI - plant health decision support

Upload (or scan) a leaf → crop + disease, confidence, Grad-CAM, lesion segmentation, severity %, environmental risk,
7/14-day progression projection, recommendations, plant history, and a Groq-powered assistant that explains the results.

## Quick start

**Step 1 - Train and save the model (notebook)**
1. Get the dataset: <https://www.kaggle.com/datasets/emmarex/plantdisease>
2. Open `notebooks/plantguard_training.ipynb`.
   * **Kaggle:** attach the dataset, enable *GPU* and *Internet*, *Run All*. When it finishes, download
     `plantguard_model_files.zip` from the Output tab and **unzip it into `model_weights/`**.
   * **Local:** extract the dataset into `datasets/` (or set env var `DATA_ROOT`), start Jupyter from the `notebooks/`
     folder and *Run All*. Files are written to `model_weights/` automatically.
   * First do a smoke test with `QUICK_TEST = True` (a few minutes), then set it to `False` for the real run.

After a real run `model_weights/` contains: `plantguard_model.keras`, `class_names.json`, `calibration.json`,
`metrics.json`, `confusion_matrix.png`, `training_curves.png` (+ split CSVs and checkpoints).

**Step 2 - Run the app**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # optional: add GROQ_API_KEY for the chatbot
uvicorn app.main:app --reload
```
Open <http://127.0.0.1:8000>, upload a leaf photo (or press **Scan Leaf**), optionally enter temperature/humidity/rain, press **Analyze**.
If the model files are missing, the page shows a banner telling you so - nothing is faked.

> Use the same TensorFlow version for training and serving (Kaggle's version is printed at the top of the notebook).
> If loading fails with a Keras version error, `pip install` the version printed there. Camera scanning needs `localhost` or HTTPS.

## What is real ML, and what is not

| Module | Method | Trained on data? |
|---|---|---|
| Image quality gate | OpenCV: sharpness, brightness, leaf-area, single-leaf checks | No (thresholds in `app/vision.py`, tune on your photos) |
| Crop + disease | EfficientNetB0/MobileNetV2 (best chosen on validation macro-F1), temperature-calibrated | **Yes** - PlantVillage |
| Unknown / low confidence | Refuses below the calibrated confidence threshold | Threshold chosen from validation coverage table |
| Grad-CAM | Gradients of the predicted class w.r.t. the last feature map | n/a (explanation of the model) |
| Segmentation + severity | **Colour-based heuristic** (HSV), severity = lesion px / leaf px | **No** - not validated until you label images (see below) |
| Environmental risk | **Rule-based weighted model** (`app/knowledge.py`) | **No** - expert assumptions, each row needs a citation |
| Progression | **Logistic growth projection**, Monte-Carlo spread over assumed rate ranges; the rate is partly fitted to the plant's own history once ≥3 scans exist | **No** - a labelled projection, not a validated forecast |
| Prediction vs actual | Earlier forecast vs the next real scan; MAE/RMSE/R² shown only when pairs exist | Computed from your real scans |
| Recommendations | Rule engine on crop + disease + severity + risk + projection (cultural measures only, no doses) | No |
| Chatbot | Groq LLM, given only the structured JSON, instructed not to invent numbers | n/a |

There is no public dataset linking PlantVillage-style images to weather and severity over time, so the risk and progression
modules are deliberately *not* presented as trained models. Details and the parameter tables: `docs/METHODS.md`.

## Project layout
```
app/            FastAPI backend (main.py, pipeline.py, vision.py, risk.py, recommend.py, knowledge.py, chat.py, db.py)
app/static/     Dashboard (index.html - upload, camera scan, charts, history, evaluation, chat)
notebooks/      plantguard_training.ipynb  (train + evaluate + save model)
model_weights/  trained model + metrics (created by the notebook)
data/           SQLite database and saved analysis images (created at runtime)
tests/          python tests/test_logic.py
docs/           METHODS.md
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | multipart: `image`, `plant_id`, `temperature`, `humidity`, `rainfall_mm`, `recent_rain`, `observed_at` → full report JSON. `status` is `ok`, `low_quality`, `no_leaf`, `multiple_leaves`, `low_confidence` or `invalid_image`; failures contain **no disease**. |
| `POST /api/quality-check` | quality gate only |
| `GET/POST /api/plants`, `GET /api/plants/{id}/history` | plants, scan history, forecast evaluation |
| `GET /api/analyses/{id}` | stored report |
| `POST /api/chat` | `{analysis_id, message, history}` → assistant reply (needs `GROQ_API_KEY`) |
| `GET /api/metrics`, `GET /api/health` | evaluation results; model status |

Interactive docs: <http://127.0.0.1:8000/docs>. API keys live only in `.env` on the server.

## Limitations - read before presenting
* **Domain shift:** PlantVillage photos are lab-style (single leaf, plain background). Phone photos in the field usually score lower.
  Test with your own photos and the PlantDoc dataset, report the drop honestly, and add field images to training if needed.
* **Only 3 crops / 15 classes** (pepper, potato, tomato). Other plants are rejected only via low confidence.
* **Segmentation is unvalidated.** Hand-label ~50 leaves (LabelMe/CVAT), compute IoU/Dice against `vision.segment`, and report it. It works best on one leaf on a plain background; viruses/mites/yellowing are measured as colour change, not lesions.
* **Severity bands** (≤10 / ≤30 / ≤60 / >60 %) are illustrative, not a published standard (`app/vision.py`).
* **Risk/progression parameters** are assumptions: add citations in `docs/METHODS.md` and adjust `app/knowledge.py`.
* Grad-CAM shows what influenced the model, not biological causality. Not a substitute for laboratory diagnosis.

## Not included yet (good next steps)
Weather-API lookup (e.g. Open-Meteo) behind a backend service, segmentation IoU/Dice report, PlantDoc/field-image evaluation,
a trained U-Net, ONNX export, PostgreSQL, Next.js frontend.
