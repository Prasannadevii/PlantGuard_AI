"""PlantGuard AI - FastAPI backend.   Run from the project root:  uvicorn app.main:app --reload"""
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, vision, chat
from .config import DATA_DIR, WEIGHTS_DIR
from .pipeline import ModelService, run_analysis
from .risk import evaluate_forecasts

STATIC = Path(__file__).parent / "static"
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_UPLOAD = 15 * 1024 * 1024

app = FastAPI(title="PlantGuard AI")
db.init()
service = ModelService()          # loads model_weights/plantguard_model.keras if present
if not db.list_plants():
    db.create_plant("My first plant")


def _parse_observed(s: Optional[str]) -> float:
    """'YYYY-MM-DD' (or ISO datetime) -> epoch seconds. Today's date means 'now'."""
    if not s:
        return time.time()
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        raise HTTPException(400, "observed_at must look like 2026-10-09")
    if d.date() == date.today() and d.hour == 0 and d.minute == 0:
        return time.time()
    if d.hour == 0 and d.minute == 0:
        d = d.replace(hour=12)
    return d.timestamp()


@app.get("/api/health")
def health():
    return {"model_ready": service.ready, "error": service.error, "model_version": service.version,
            "classes": len(service.class_names)}


class PlantIn(BaseModel):
    name: str
    crop: Optional[str] = None
    location: Optional[str] = None


@app.get("/api/plants")
def plants():
    return db.list_plants()


@app.post("/api/plants")
def new_plant(p: PlantIn):
    return db.get_plant(db.create_plant(p.name, p.crop, p.location))


@app.post("/api/quality-check")
def quality_check(image: UploadFile = File(...)):
    bgr = vision.decode_image(image.file.read(MAX_UPLOAD))
    if bgr is None:
        raise HTTPException(400, "Could not read the file as an image.")
    return vision.quality_check(bgr)


@app.post("/api/analyze")
def analyze(
    image: UploadFile = File(...),
    plant_id: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    humidity: Optional[float] = Form(None),
    rainfall_mm: Optional[float] = Form(None),
    recent_rain: bool = Form(False),
    observed_at: Optional[str] = Form(None),
):
    if not service.ready:
        raise HTTPException(503, f"Model not loaded. {service.error}")
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported file type. Use JPG, PNG or WebP.")
    data = image.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 15 MB.")
    if humidity is not None and not 0 <= humidity <= 100:
        raise HTTPException(400, "Humidity must be between 0 and 100.")
    if plant_id is None:
        plant_id = db.list_plants()[0]["id"]
    if not db.get_plant(plant_id):
        raise HTTPException(404, "Unknown plant_id")

    obs = _parse_observed(observed_at)
    env = {"temperature": temperature, "humidity": humidity, "rainfall_mm": rainfall_mm,
           "recent_rain": recent_rain, "observed_at": obs}
    report, images = run_analysis(service, data, env, db.history(plant_id))
    if report["status"] != "ok":
        return JSONResponse(report)                      # no diagnosis is saved or shown

    d = report
    aid = db.save_analysis(plant_id, obs, dict(
        class_name=d["disease"]["class_name"], crop=d["crop"]["name"], disease=d["disease"]["name"],
        disease_key=d["disease"]["key"] or "healthy", confidence=d["disease"]["confidence"],
        severity_pct=d["severity"]["affected_percent"], severity_level=d["severity"]["level"],
        temperature=temperature, humidity=humidity, rainfall=rainfall_mm, recent_rain=recent_rain,
        risk_score=(d["risk"] or {}).get("score"), risk_level=(d["risk"] or {}).get("level"),
        curve=(d["progression"] or {}).get("curve"), model_version=d["model_version"]))
    folder = DATA_DIR / "analyses" / str(aid)
    folder.mkdir(parents=True, exist_ok=True)
    urls = {}
    for name, img in images.items():
        (folder / f"{name}.jpg").write_bytes(vision.jpg_bytes(img))
        urls[name] = f"/files/analyses/{aid}/{name}.jpg"
    d.update({"analysis_id": aid, "plant_id": plant_id, "observed_at": obs, "images": urls})
    db.update_report(aid, d)
    return d


@app.get("/api/analyses/{aid}")
def get_analysis(aid: int):
    r = db.get_analysis(aid)
    if not r:
        raise HTTPException(404, "Analysis not found")
    return r


@app.get("/api/plants/{pid}/history")
def plant_history(pid: int):
    if not db.get_plant(pid):
        raise HTTPException(404, "Unknown plant")
    rows = db.history(pid)
    scans = [{k: r[k] for k in ("id", "observed_at", "disease", "disease_key", "confidence", "severity_pct",
                                "severity_level", "risk_score", "risk_level", "temperature", "humidity", "rainfall")}
             for r in rows]
    ev = evaluate_forecasts([{"id": r["id"], "t": r["observed_at"], "severity": r["severity_pct"], "curve": r["curve"]}
                             for r in rows if r["disease_key"] != "healthy"])
    return {"plant": db.get_plant(pid), "scans": scans, "forecast_evaluation": ev}


class ChatIn(BaseModel):
    analysis_id: int
    message: str
    history: list = []


@app.post("/api/chat")
def chat_endpoint(c: ChatIn):
    report = db.get_analysis(c.analysis_id)
    if not report:
        raise HTTPException(404, "Analysis not found")
    return chat.ask(report, c.message, c.history)


@app.get("/api/metrics")
def metrics():
    f = WEIGHTS_DIR / "metrics.json"
    if not f.exists():
        return {"available": False}
    return {"available": True, **json.loads(f.read_text())}


@app.get("/api/metrics/figure/{name}")
def metric_figure(name: str):
    if name not in ("confusion_matrix.png", "training_curves.png"):
        raise HTTPException(404)
    f = WEIGHTS_DIR / name
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
