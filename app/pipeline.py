"""End-to-end analysis pipeline.

ModelService is the ONLY place that touches TensorFlow (imported lazily), so everything else can be
tested without it. run_analysis() accepts any object with the same interface (see tests/FakeService).
"""
from __future__ import annotations
import json
import time
import numpy as np
import cv2

from . import vision, risk as riskmod
from .config import WEIGHTS_DIR, MODEL_FILE, LOW_CONF
from .knowledge import parse_class
from .recommend import recommend


class ModelService:
    """Loads the notebook-trained Keras model and runs prediction + Grad-CAM in one forward pass."""

    def __init__(self, weights_dir=WEIGHTS_DIR):
        self.ready, self.error = False, None
        self.class_names, self.temperature, self.low_conf, self.version = [], 1.0, LOW_CONF, "unknown"
        try:
            model_path = weights_dir / MODEL_FILE
            names_path = weights_dir / "class_names.json"
            if not model_path.exists() or not names_path.exists():
                raise FileNotFoundError(
                    f"No trained model found in {weights_dir}. Run notebooks/plantguard_training.ipynb first "
                    f"and copy {MODEL_FILE}, class_names.json (and the other output files) into that folder.")
            import tensorflow as tf  # lazy: heavy import
            self.tf = tf
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.class_names = json.loads(names_path.read_text())
            cal = weights_dir / "calibration.json"
            if cal.exists():
                c = json.loads(cal.read_text())
                self.temperature = float(c.get("temperature", 1.0))
                self.low_conf = float(c.get("low_confidence_threshold", LOW_CONF))
            self.version = f"{self.model.name}-{time.strftime('%Y%m%d', time.localtime(model_path.stat().st_mtime))}"
            self.grad_model = tf.keras.Model(
                self.model.inputs, [self.model.get_layer("gradcam_layer").output, self.model.output])
            self.ready = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def predict(self, rgb224: np.ndarray):
        """rgb224: uint8 (224,224,3). Returns (calibrated probabilities, Grad-CAM map in [0,1])."""
        tf = self.tf
        x = tf.convert_to_tensor(rgb224[None].astype("float32"))      # model expects raw 0-255 pixels
        with tf.GradientTape() as tape:
            conv, preds = self.grad_model(x, training=False)
            top = tf.argmax(preds[0])
            score = preds[:, top]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        p = preds[0].numpy().astype("float64")
        z = np.log(np.clip(p, 1e-12, 1)) / self.temperature             # temperature scaling
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        return p, cam


FAIL_MESSAGES = {
    "no_leaf": "No valid leaf detected. Place a single leaf in the frame against a plain background.",
    "multiple_leaves": "Several leaves detected. Please capture one clear leaf.",
    "low_quality": "Image quality is insufficient for a reliable analysis.",
    "low_confidence": "Prediction confidence is low, so no diagnosis is shown. Please capture a clearer image of one leaf "
                      "(the plant may also be outside the 3 crops this model knows: pepper, potato, tomato).",
    "invalid_image": "The file could not be read as an image.",
}

LIMITATIONS = [
    "Model trained on PlantVillage (lab-style images); accuracy on phone photos in the field is usually lower.",
    "Severity comes from a colour-based segmentation heuristic that has not been validated against expert masks.",
    "Environmental risk is a rule-based model; the progression is a scenario projection, not a validated forecast.",
    "Grad-CAM shows which regions influenced the model, not biological causality.",
]


def run_analysis(service, image_bytes, env=None, history_rows=None):
    """Returns (report dict, images dict name->BGR array). report['status'] != 'ok' means no diagnosis."""
    env = env or {}
    bgr = vision.decode_image(image_bytes)
    if bgr is None:
        return {"status": "invalid_image", "message": FAIL_MESSAGES["invalid_image"]}, {}
    q = vision.quality_check(bgr)
    if q["status"] != "ok":
        return {"status": q["status"], "message": FAIL_MESSAGES[q["status"]], "quality": q}, {}

    disp = vision.prepare_leaf(bgr)                                      # 384x384 BGR
    rgb224 = cv2.cvtColor(cv2.resize(disp, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    probs, cam = service.predict(rgb224)
    order = np.argsort(probs)[::-1]
    infos = [parse_class(n) for n in service.class_names]
    top3 = [{"label": f"{infos[i][0]} - {infos[i][1]}", "probability": round(100 * float(probs[i]), 1)} for i in order[:3]]
    best = int(order[0])
    conf = float(probs[best])
    if conf < service.low_conf:
        return {"status": "low_confidence", "message": FAIL_MESSAGES["low_confidence"], "quality": q,
                "candidates": top3}, {}

    crop, disease, key = infos[best]
    crop_conf = float(sum(p for p, inf in zip(probs, infos) if inf[0] == crop))

    heat, cam_overlay = vision.heat_images(disp, cam)
    images = {"original": disp, "heatmap": heat, "cam_overlay": cam_overlay}

    seg = vision.segment(disp)
    if key is None:                                                      # healthy: lesion analysis not applicable
        sev = {"applicable": False, "affected_percent": 0.0, "healthy_percent": 100.0, "level": "None",
               "leaf_pixels": seg["leaf_pixels"], "diseased_pixels": 0,
               "method": "Classifier reports a healthy leaf, so no lesion area is measured."}
        les = np.zeros_like(seg["lesion"])
        sev_pct = 0.0
    else:
        sev_pct = round(seg["percent"], 1)
        les = seg["lesion"]
        sev = {"applicable": True, "affected_percent": sev_pct, "healthy_percent": round(100 - sev_pct, 1),
               "level": vision.severity_level(sev_pct), "leaf_pixels": seg["leaf_pixels"],
               "diseased_pixels": seg["lesion_pixels"],
               "method": "Severity = lesion pixels / leaf pixels x 100 (colour-based segmentation heuristic)",
               "bands": "<=10% Low, <=30% Moderate, <=60% High, >60% Critical (illustrative bands, adjustable in code)"}
    mask_img, seg_overlay = vision.seg_images(disp, seg["leaf"], les)
    images.update({"mask": mask_img, "seg_overlay": seg_overlay})

    environment = {k: env.get(k) for k in ("temperature", "humidity", "rainfall_mm")}
    environment["recent_rain"] = bool(env.get("recent_rain"))
    risk = progression = None
    if key is not None:
        risk = riskmod.assess_risk(key, sev_pct, env.get("temperature"), env.get("humidity"),
                                   env.get("rainfall_mm"), environment["recent_rain"])
        pts = [(0.0, sev_pct)]
        now_t = env.get("observed_at")
        if now_t is not None:
            for r in history_rows or []:
                if r.get("disease_key") == key and r.get("observed_at") is not None:
                    pts.append(((r["observed_at"] - now_t) / 86400.0, r["severity_pct"]))
        progression = riskmod.forecast(key, sev_pct, risk["score"] if env.get("temperature") is not None
                                       or env.get("humidity") is not None else None, sorted(pts))
    rec = recommend(crop, disease, key, sev_pct, sev["level"], risk, progression)

    report = {
        "status": "ok", "model_version": getattr(service, "version", "unknown"), "quality": q,
        "crop": {"name": crop, "confidence": round(100 * crop_conf, 1)},
        "disease": {"name": disease, "class_name": service.class_names[best], "key": key,
                    "confidence": round(100 * conf, 1), "is_healthy": key is None},
        "top3": top3, "severity": sev, "environment": environment,
        "risk": risk, "progression": progression, "recommendation": rec, "limitations": LIMITATIONS,
    }
    return report, images
