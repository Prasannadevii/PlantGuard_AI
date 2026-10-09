# PlantGuard AI - Developer Guide

For anyone who needs to understand, debug, tune or extend the project. For installation and usage see `README.md`;
for formulas and parameter tables see `docs/METHODS.md`.

## Contents
1. [Design principles](#1-design-principles)
2. [Architecture and request flow](#2-architecture-and-request-flow)
3. [The model contract](#3-the-model-contract)
4. [Module reference](#4-module-reference)
5. [Data model and storage](#5-data-model-and-storage)
6. [Training notebook internals](#6-training-notebook-internals)
7. [Frontend internals](#7-frontend-internals)
8. [Common tasks](#8-common-tasks)
9. [Validating the heuristic modules](#9-validating-the-heuristic-modules)
10. [Testing strategy](#10-testing-strategy)
11. [Debugging recipes](#11-debugging-recipes)
12. [Deployment notes](#12-deployment-notes)
13. [Conventions and pitfalls](#13-conventions-and-pitfalls)
14. [Pre-submission checklist](#14-pre-submission-checklist)

---

## 1. Design principles

1. **Honest ML.** No hard-coded results. Every number the UI shows is computed from the image, the inputs, or
   `metrics.json`. Modules that are not trained models (segmentation, risk, progression) are labelled as such in the
   API output and the UI.
2. **Never force a diagnosis.** Quality failures and low confidence return a status and *no* `disease` field.
3. **Visual evidence for every output.** Each report section pairs a prediction with its evidence (chart, image, factor table).
4. **TensorFlow is isolated.** Only `ModelService` in `app/pipeline.py` imports it (lazily). Everything else runs
   without it, which keeps tests fast and lets the logic be tested with a fake model.
5. **LLM explains, never computes.** The Groq assistant receives only the structured report and is instructed not to
   invent numbers.
6. **Secrets stay server-side.** Keys come from environment variables; the browser only talks to `/api/*`.

---

## 2. Architecture and request flow

```
Browser (app/static/index.html)
   │  multipart POST /api/analyze
   ▼
app/main.py ── validates type/size/inputs ── loads plant history from SQLite
   │
   ▼
app/pipeline.run_analysis(service, bytes, env, history)
   1. vision.decode_image        EXIF-rotate, cap at 1600 px
   2. vision.quality_check       → status ok | no_leaf | multiple_leaves | low_quality
   3. vision.prepare_leaf        square crop around main leaf → 384×384 BGR
   4. resize → 224×224 RGB       cv2.INTER_AREA
   5. service.predict            calibrated probabilities + Grad-CAM map
   6. low-confidence check       calibrated max prob < service.low_conf → no diagnosis
   7. vision.heat_images         heatmap + overlay
   8. vision.segment             leaf mask, lesion mask, severity %
   9. risk.assess_risk           rule-based score + factor breakdown
  10. risk.forecast              logistic projection, days 0-14
  11. recommend.recommend        priority, actions, evidence
   │
   ▼
main.py saves row + 5 JPEGs, returns report JSON ──► dashboard renders
```

Endpoints are plain `def` (not `async def`) so FastAPI runs the CPU-heavy work in its thread pool instead of
blocking the event loop.

---

## 3. The model contract

The web app is architecture-agnostic as long as the saved model honours this contract (the notebook's three builders do):

| Item | Requirement |
|---|---|
| File | `model_weights/plantguard_model.keras` (Keras 3 native format) |
| Input | `float32`, shape `(N, 224, 224, 3)`, **raw RGB pixel values 0-255** (rescaling/normalisation is inside the model) |
| Output | softmax over classes, shape `(N, num_classes)`, order = `class_names.json` |
| Special layer | a layer named **`gradcam_layer`** whose output is the last spatial feature map (e.g. 7×7×1280) |
| Augmentation | inside the model; inactive at inference |
| `class_names.json` | JSON list of folder names, in label order |
| `calibration.json` | `{"temperature": T, "low_confidence_threshold": 0.60}` (optional; defaults T=1, 0.60) |

Calibration is applied to the output probabilities as `softmax(log(p) / T)`. Because softmax is invariant to a
constant shift, this is equivalent to temperature-scaling the logits.

If you swap in a new backbone, keep the contract and nothing else changes. If you change the input size, update
`IMG_SIZE` in the notebook **and** the 224 constants in `app/pipeline.py`.

---

## 4. Module reference

### `app/config.py`
Paths and environment variables (`WEIGHTS_DIR`, `DATA_DIR`, `DB_PATH`, `LOW_CONF`, Groq settings). Loads `.env`
if `python-dotenv` is installed.

### `app/knowledge.py` - domain data, no logic
- `CLASS_INFO`: folder name → `(crop, disease label, disease key)`; key `None` means *healthy*. Unknown folder names fall back to `parse_class()` with key `"generic"`.
- `RISK_PROFILES`: per-disease temperature trapezoid, humidity ramp, rain reference, weights `(T, H, R, severity)`, `inverse` flag for dry-favoured problems.
- `GROWTH_RATES`: per-disease logistic rate range (per day) at risk 0 → 100.
- `ACTIONS`: cultural/sanitation measures per disease; `DISCLAIMER`.
All values are **assumptions** - cite or correct them.

### `app/vision.py` - classical CV (OpenCV + NumPy only)
| Function | Purpose |
|---|---|
| `decode_image(bytes)` | bytes → BGR (EXIF-rotated, ≤1600 px) or `None` |
| `leaf_mask(bgr)` | vegetation-coloured region (HSV hue 8-100, S≥35, V≥25), open/close, external contours ≥2 % of the frame filled |
| `quality_check(bgr)` | returns `status`, `score`, `checks`, `tips`, `metrics` |
| `prepare_leaf(bgr, out=384)` | square crop around the largest leaf (+16 % margin); square leaf-filled images are left untouched |
| `segment(bgr)` | eroded leaf mask, "healthy green" test (hue 36-95, S≥40, V≥35), lesion = leaf ∖ healthy, opened + median filtered; <0.5 % treated as 0 |
| `seg_images`, `heat_images` | mask/overlay and heatmap/overlay pictures |
| `severity_level(pct)` | ≤10 Low, ≤30 Moderate, ≤60 High, else Critical |

Constants at the top: `BLUR_FAIL=15`, `BLUR_WARN=40` (variance of the Laplacian), `DARK_FAIL=35`, `BRIGHT_FAIL=235`,
`MIN_LEAF_FRAC=0.08`, `SEVERITY_BANDS`. Quality `status` priority: no leaf → multiple leaves → low quality.

### `app/pipeline.py`
- `ModelService`: loads the model, `class_names`, calibration; builds `grad_model` (inputs → [`gradcam_layer` output, predictions]). `predict(rgb224)` runs **one** forward pass inside a `GradientTape` and returns `(calibrated probs, cam in [0,1])`. If loading fails, `ready=False` and `error` explains why (the app still starts).
- `run_analysis(service, image_bytes, env, history_rows)` → `(report, images)`. `service` can be anything exposing `class_names`, `low_conf`, `version`, `predict()` - tests use a fake.

### `app/risk.py`
- `assess_risk(key, severity, temperature, humidity, rainfall, recent_rain)`: factor favourability 0-1 × weight; missing inputs → 0.5 and `env_complete=False`. Levels: <35 Low, ≤65 Moderate, else High.
- `forecast(key, severity, risk_score, history, n=2000, seed=0)`: logistic growth `S(t)=K/(1+((K−S0)/S0)e^(−rt))`; `r` from the rate range scaled by risk, log-normal noise (σ 0.40, or 0.55 when no environment given); `K~U(70,100)`; seeded → reproducible. With ≥3 history points, `r` is blended with the fitted slope of `logit(S/100)` vs time using weight `n/(n+3)`. Returns `curve` (15 points with median/low/high), `milestones`, `source`, `uncertainty_note`.
- `evaluate_forecasts(rows)`: pairs each stored forecast with the **next** scan 0.5-14 days later; returns pairs, MAE, RMSE, R² (R² only with ≥3 pairs and varying actuals).

### `app/recommend.py`
Priority points = severity (0-3) + risk (0-2) + projected day-7 severity (<20 → 0, <45 → 1, else 2); ≤2 LOW, 3-4 MEDIUM, ≥5 HIGH.
Returns actions, next-scan interval (HIGH 2-3, MEDIUM 3-5, LOW 5-7 days), an evidence table that spells out the rule, and a disclaimer.

### `app/chat.py`
`compact_context(report)` strips URLs/images; `ask(report, message, history)` calls Groq's OpenAI-compatible
chat-completions endpoint (temperature 0.2, last 6 turns, input length caps). Returns `{configured, reply}`; never raises.

### `app/db.py`
Thin SQLite wrapper (one short-lived connection per call). See §5.

### `app/main.py`
Routes, input validation, saving images/rows. Failure statuses are returned with HTTP 200 as JSON bodies (they are valid
outcomes, not server errors); hard errors use 4xx/5xx.

---

## 5. Data model and storage

SQLite file `data/plantguard.db`, created on first start.

```
plants(id, name, crop, location, created_at)

analyses(id, plant_id → plants.id, observed_at, created_at,
         class_name, crop, disease, disease_key, confidence,
         severity_pct, severity_level,
         temperature, humidity, rainfall, recent_rain,
         risk_score, risk_level,
         curve_json,      -- 15 forecast points; used for predicted-vs-actual
         report_json,     -- full report returned to the client
         model_version)
```

- `observed_at` (epoch seconds) is the *scan date* - back-dating is allowed and is what history/forecast logic uses; `created_at` is when the row was written.
- Healthy scans are stored with `disease_key = "healthy"`; they are excluded from forecast evaluation.
- Images: `data/analyses/<id>/{original,heatmap,cam_overlay,mask,seg_overlay}.jpg`, served under `/files/...`.
- Only `status == "ok"` analyses are saved.

To move to PostgreSQL, re-implement the functions in `db.py` with the same signatures (psycopg/SQLAlchemy); nothing else imports sqlite.

---

## 6. Training notebook internals

`notebooks/plantguard_training.ipynb`, 10 code cells, runs top to bottom.

| Cell | Does | Key variables |
|---|---|---|
| 1 | Settings, imports, dataset discovery | `QUICK_TEST`, `USE_PRETRAINED`, `GROUPED_SPLIT`, `EPOCHS_HEAD`, `EPOCHS_FINETUNE`, `FINETUNE_LAYERS`, `LOW_CONF_THRESHOLD`, `OUT_DIR` |
| 2 | Cleaning + dataset report | `df`, `dataset_report` |
| 3 | Near-duplicate grouping + split | `MAX_GROUP=40`, `train_df/val_df/test_df` |
| 4 | tf.data pipelines, augmentation, class weights | `class_names`, `class_weight` |
| 5 | `build_model(kind)`, `fit_model`, `macro_f1` | contract in §3 |
| 6 | Train 3 models, compare on validation | `comparison`, `best_kind` |
| 7 | Fine-tune best, curves, overfit gap | `final_model`, `hist` |
| 8 | Test evaluation, confusion matrix | `acc`, `f1`, `per_class`, `cm` |
| 9 | Temperature scaling, ECE, coverage table | `T`, `coverage_rows` |
| 10 | Save + reload verification + `metrics.json` | `OUT_DIR/*` |

Notes:
- **Grouped split.** `V` holds, for every image, 8 normalised 16×16 fingerprints (4 rotations × optional flip). Per class, cosine similarity between each image and all orientations of all others builds a "same leaf" graph; connected components are the groups. The threshold is tried from 0.97 to 0.995 and the first one whose largest group ≤ `MAX_GROUP` is used. If a threshold yields >3 M matching pairs it is rejected as too loose. If none works the cell falls back to one group per image and prints a warning (leakage possible). Splitting uses `StratifiedGroupKFold(20)` → folds 0-13 train, 14-16 val, 17-19 test. Tune `MAX_GROUP` / the threshold list if the printed statistics look off (e.g. one giant group = threshold too loose; zero groups with >1 image on a dataset known to contain copies = too strict).
- **Selection discipline.** Models are compared on validation only; the test set is touched once in cell 8. Do not re-run cell 8 after changing hyper-parameters - restart from the top instead.
- **Fine-tuning** unfreezes the last `FINETUNE_LAYERS` of the backbone but keeps all BatchNormalization layers frozen; weights are restored if validation macro-F1 does not improve.
- **Class weights** are the square root of sklearn's "balanced" weights.
- Checkpoints go to `model_weights/checkpoints/` (best validation accuracy per stage).
- To add a model, add a branch in `build_model` that keeps the contract and add its name to `KINDS`.
- Quick smoke test: `QUICK_TEST = True` (60 images/class, 1 epoch). Its metrics are flagged in `metrics.json` and the UI.

`metrics.json` keys read by the dashboard: `test`, `selected_model`, `fine_tuned`, `quick_test`, `comparison_validation`,
`per_class`, `overfitting`, `calibration`, `dataset.grouped_split`, `generated`, `figures`.

---

## 7. Frontend internals

Single file `app/static/index.html` (vanilla JS, Chart.js from cdnjs; if Chart.js fails to load the charts are skipped and
everything else still works).

| Function | Role |
|---|---|
| `init()` | `/api/health` banner, load plants, load evaluation |
| `setFile(file)` | shared by upload and camera capture; enables **Analyze** |
| camera: `#camBtn` handler, `hint()`, `#shot` | `getUserMedia` (rear camera preferred), 700 ms loop sampling a 160×120 frame for brightness / blur / leaf-in-box hints, capture → JPEG `File`. Hints are advisory; the **server quality check is authoritative** |
| `#go` handler | builds `FormData`, posts to `/api/analyze`, routes to `fail()` or `render()` |
| `render(report)` | builds all sections from the JSON; `mk()` creates/destroys Chart.js charts by canvas id |
| `fail(report)` | shows message, failed checks, tips, and "closest guesses (not a diagnosis)" |
| `loadHist()` | history table, timeline chart, predicted-vs-actual |
| `loadEval()` | evaluation section from `/api/metrics` |
| `ask()` | chat; sends `analysis_id`, message, last turns |

Rules: every server string is inserted through `esc()`; chat replies use `textContent`. The page expects the report
shape in README §9 - if you change a field name in `pipeline.py`, update `render()`.

---

## 8. Common tasks

### Tune the quality gate
Edit the constants in `app/vision.py`. Use `POST /api/quality-check` with a folder of your own good and bad photos
and print `metrics` (`sharpness`, `brightness`, `leaf_fraction`) to pick thresholds from real distributions
rather than guessing.

### Change severity bands
`SEVERITY_BANDS` in `app/vision.py`. If you want disease-specific bands, replace `severity_level(pct)` with a
function taking the disease key and look the bands up in `knowledge.py`; pass the key from `pipeline.run_analysis`.

### Add or change a disease / crop
1. Retrain with the new class folder (the notebook discovers classes by folder name; the crop filter in cells 1-2 uses the `CROPS` list - extend it).
2. In `app/knowledge.py` add the folder name to `CLASS_INFO` (otherwise it falls back to `generic`), then add the disease key to `RISK_PROFILES`, `GROWTH_RATES` and `ACTIONS`.
3. Add a test in `tests/test_logic.py` mirroring `test_risk_direction`.
4. Copy the new `model_weights/` files and restart.

### Replace the rule-based risk with a trained model
Keep `assess_risk`'s return shape (`score`, `level`, `method`, `factors`, `env_complete`) so the UI, recommendations and chat
keep working. Set `method` to the truth (e.g. "Gradient-boosting model trained on <dataset>") and report its MAE/RMSE.
Only do this if you have real labelled environmental/disease data - otherwise you would be training on your own assumptions.

### Replace the progression projection
Same approach: keep the `forecast()` return shape (`curve`, `milestones`, `source`, `uncertainty_note`). If you obtain real
time-series severity data, a mixed-effects growth model or gradient-boosted regressor with quantile loss can supply
the median/low/high; evaluate with `evaluate_forecasts` on held-out plants.

### Add live weather (sketch)
Add a backend route that fetches current conditions server-side so no key or CORS issue reaches the browser, e.g. with
Open-Meteo (no API key; verify the parameter names in its documentation):

```python
# app/weather.py
import requests
def current(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation"})
    r.raise_for_status()
    c = r.json()["current"]
    return {"temperature": c["temperature_2m"], "humidity": c["relative_humidity_2m"], "rainfall_mm": c["precipitation"]}
```
Expose it as `GET /api/weather?lat=&lon=` and add a "Use my location" button that fills the three inputs. Note that
`precipitation` is the current interval's amount; for "recent rain" you would also want the past-24 h total (hourly data).

### Swap the segmenter for a trained model
Replace `vision.segment()` with a function returning the same dict (`leaf`, `lesion`, `leaf_pixels`, `lesion_pixels`, `percent`
with 0/1 masks at 384×384). `pipeline.py` and the UI need no other change. Train on your hand-labelled masks (plus augmentation) and
report IoU/Dice on a held-out subset.

### Change the Groq model or prompt
`GROQ_MODEL` env var; the system prompt is `SYSTEM` in `app/chat.py`. Keep the rules "use only numbers in the JSON" and "no
doses/products". Extend `compact_context()` if the assistant needs another field.

### Use a different port / host
`uvicorn app.main:app --host 0.0.0.0 --port 8080`. Camera access then requires HTTPS (except on localhost).

---

## 9. Validating the heuristic modules

The segmentation/severity is a colour heuristic; **measure it** before presenting severity as reliable.

1. Run ~50 varied leaves through the app. For each, take `data/analyses/<id>/original.jpg` (this is the exact 384×384 crop the segmenter sees, so masks align).
2. Annotate the lesion area in LabelMe/CVAT and export a binary mask `<id>.png` (255 = lesion).
3. Compute metrics:

```python
import cv2, glob, numpy as np
from app import vision

def iou_dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter, union = (p & g).sum(), (p | g).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
    return iou, dice

rows = []
for img_path in sorted(glob.glob("labels/*.jpg")):
    gt = cv2.imread(img_path.replace(".jpg", ".png"), 0) > 127
    seg = vision.segment(cv2.imread(img_path))
    iou, dice = iou_dice(seg["lesion"], gt)
    sev_gt = 100 * gt.sum() / max(seg["leaf_pixels"], 1)
    rows.append((iou, dice, seg["percent"], sev_gt))
r = np.array(rows)
print("mean IoU %.3f  mean Dice %.3f  severity MAE %.1f points" % (r[:, 0].mean(), r[:, 1].mean(), np.abs(r[:, 2] - r[:, 3]).mean()))
```
4. Report the numbers (and the failure cases) in your documentation. Adjust the HSV thresholds in `segment()` or move to a trained U-Net if they are poor.

Other validation to do and report:
- **Domain shift:** evaluate the saved model on your own phone photos and PlantDoc; compare with PlantVillage test accuracy.
- **Quality gate:** confusion between your "good" and "bad" photos and the gate's decisions.
- **Progression:** `evaluate_forecasts` after collecting real repeat scans of the same plants.

---

## 10. Testing strategy

`python tests/test_logic.py` (also works with pytest). No TensorFlow/FastAPI needed.

| Test | Verifies |
|---|---|
| `test_severity_levels` | band boundaries (10 → Low, 60 → High, >60 → Critical) |
| `test_healthy_leaf_has_zero_severity`, `test_spots_increase_severity_and_are_measured` | severity arithmetic on synthetic leaves with known spot area |
| `test_quality_gate` | ok / dark / blurry / no-leaf / two-leaf |
| `test_prepare_leaf_crops_offcentre_leaf` | crop enlarges an off-centre leaf |
| `test_risk_direction` | wet+warm > cold+dry; mites invert; missing env flagged; contributions sum to the score |
| `test_forecast_properties`, `test_forecast_uses_history` | day-0 equals measured, monotone, bounded 0-100, ordered band, higher risk → higher projection, reproducible, history blending |
| `test_forecast_evaluation` | pairing logic; R² needs ≥3 pairs |
| `test_recommendation_priority` | HIGH/LOW/healthy cases |
| `test_db_roundtrip` | storage and history |
| `test_pipeline_ok_lowconf_healthy_and_failures` | whole pipeline with `FakeService`: ok, low confidence, healthy, invalid image, no leaf |

**Writing tests that need a model:** instantiate a fake service
(`class_names`, `low_conf`, `version`, `predict(rgb) -> (probs, cam)`); see `FakeService` in the test file. For API-level tests install
FastAPI and use `fastapi.testclient.TestClient(app.main.app)` after replacing `app.main.service` with the fake (give it `ready=True`).

What the tests do **not** cover: real model accuracy, real segmentation accuracy, the browser UI, Groq. Those need the notebook's
`metrics.json`, the §9 validation, a manual browser pass and a live key respectively.

---

## 11. Debugging recipes

| Question | How |
|---|---|
| Why was my image rejected? | `POST /api/quality-check` and read `checks` and `metrics` |
| What does the segmenter see? | `cv2.imwrite("leaf.png", vision.leaf_mask(img)[0]*255)`; `s = vision.segment(img)` then write `s["leaf"]*255` / `s["lesion"]*255` |
| Is the crop sane? | open `data/analyses/<id>/original.jpg` - this is both the model's view and the segmenter's view |
| Prediction looks off | confirm the model contract (0-255 input); compare `class_names.json` with the notebook's class list; check `calibration.json` temperature isn't extreme (T≫3 means a poorly calibrated or overfit model) |
| Heatmap looks like noise | check that `gradcam_layer` exists in the loaded model and that the nested backbone was saved intact (notebook cell 10 asserts this) |
| Forecast looks flat/odd | print `risk.forecast(key, sev, risk)["growth_rate_per_day"]`; compare to `GROWTH_RATES`; check the `source` string to see whether history was blended |
| Predicted-vs-actual empty | needs a later scan 0.5-14 days after an earlier one with a stored curve; healthy scans are excluded |
| App starts but analyze returns 503 | `GET /api/health` → `error` field has the cause (missing files, TensorFlow import, version mismatch) |

---

## 12. Deployment notes

- Run with a **single worker** per container/VM (`uvicorn app.main:app --workers 1`): each worker loads the full TensorFlow model into memory. Scale by running more containers behind a load balancer, not more workers, unless you have the RAM.
- First request after start is slower (TensorFlow graph warm-up). You can warm up by calling `service.predict` on a blank image at startup.
- Put a reverse proxy (nginx/Caddy) in front for HTTPS (needed for camera access on non-localhost) and add authentication - the app has none.
- Persist `data/` (database + images) and `model_weights/` on volumes. Back up `plantguard.db`.
- Set `GROQ_API_KEY` as a secret/environment variable, not in the image. Never commit `.env`.
- For inference only (no notebook), install just the "web app" and "model" sections of `requirements.txt`; consider `tensorflow-cpu` on servers without GPUs.
- Consider ONNX export for lighter serving; you would then reimplement the Grad-CAM step (needs gradients) or compute it with a PyTorch/TF sidecar.

---

## 13. Conventions and pitfalls

- **Colour order.** OpenCV is BGR, the model wants RGB. `pipeline.run_analysis` converts exactly once (`cv2.COLOR_BGR2RGB` before `predict`). Overlays stay BGR for `cv2.imencode`.
- **Never claim more than the data supports.** If you add a metric to the UI, it must come from a computed result with a stated sample size.
- **Failure statuses are API contract.** The frontend branches on `status != "ok"`; keep new failure modes in the same shape (`status`, `message`, optional `quality`/`candidates`) and add a message to `FAIL_MESSAGES`.
- **Don't leak the test set.** Any change that uses test data for choices (threshold, model, augmentation) invalidates the reported test numbers - use validation.
- **TensorFlow/Keras versions.** `.keras` files are version-sensitive; record the TensorFlow version (it's in `metrics.json`) and pin it for serving.
- **Model contract changes** (input size, scaling, layer names) require changes in both the notebook and `app/pipeline.py`, plus retraining.
- **Class order.** Always load `class_names.json` from the same training run as the model.
- **Style.** Type hints where helpful, small pure functions in `vision.py`/`risk.py`, no TensorFlow imports outside `ModelService`, user-visible strings in the module that produces them.
- **Large files.** Don't commit `datasets/`, `data/`, `.env` or checkpoints (see `.gitignore`); commit `metrics.json` and the two PNGs if you want results in the repo.

---

## 14. Pre-submission checklist

- [ ] Notebook run with `QUICK_TEST = False`; `metrics.json` is from that run
- [ ] Group-leakage and duplicate checks printed **PASS**
- [ ] Test accuracy, macro-F1, per-class table, confusion matrix and train/val curves included in the report - with the overfitting gap discussed
- [ ] Evaluated on your own phone photos (and/or PlantDoc); the drop is reported honestly
- [ ] Segmentation IoU/Dice measured on ~50 hand-labelled leaves (§9)
- [ ] Every row in the `docs/METHODS.md` risk table has a source (or is clearly stated as an assumption)
- [ ] Report states plainly: risk = rule-based, progression = projection with scenario spread, not trained forecasters
- [ ] A few real repeat scans collected so the predicted-vs-actual panel shows real pairs (or is described as "insufficient data")
- [ ] Demo includes the failure cases (blurry photo, non-leaf object, two leaves)
- [ ] `.env` not committed; `python tests/test_logic.py` passes
