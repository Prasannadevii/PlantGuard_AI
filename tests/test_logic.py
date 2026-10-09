"""Logic tests that need no TensorFlow/FastAPI. Run:  python tests/test_logic.py   (or pytest)
They use SYNTHETIC leaves to check the mechanics (masks, severity arithmetic, risk/forecast behaviour,
DB, pipeline plumbing). They say nothing about real-world model accuracy."""
import os, sys, tempfile, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import vision, risk, db, recommend as rec_mod
from app.pipeline import run_analysis

BG = (170, 165, 160)  # grey background (BGR)


def synth_leaf(size=256, spots=0, seed=0):
    img = np.full((size, size, 3), BG, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, int(size / 2.6)), 20, 0, 360, (45, 150, 55), -1)
    rng = np.random.default_rng(seed)
    for _ in range(spots):
        cv2.circle(img, (size // 2 + int(rng.integers(-40, 40)), size // 2 + int(rng.integers(-50, 50))), 14, (25, 70, 120), -1)
    return img


def enc(img):
    return vision.jpg_bytes(img, 95)


def test_severity_levels():
    assert [vision.severity_level(x) for x in (0, 10, 10.1, 30, 60, 60.1)] == ["Low", "Low", "Moderate", "Moderate", "High", "Critical"]


def test_healthy_leaf_has_zero_severity():
    seg = vision.segment(synth_leaf())
    assert seg["percent"] == 0.0 and seg["leaf_pixels"] > 5000


def test_spots_increase_severity_and_are_measured():
    s0 = vision.segment(synth_leaf(spots=0))["percent"]
    s1 = vision.segment(synth_leaf(spots=3))["percent"]
    s2 = vision.segment(synth_leaf(spots=8))["percent"]
    print("severity healthy/3 spots/8 spots:", s0, round(s1, 1), round(s2, 1))
    assert s0 < s1 < s2 and 3 < s1 < 40


def test_quality_gate():
    ok = vision.quality_check(synth_leaf())
    assert ok["status"] == "ok", ok
    dark = vision.quality_check((synth_leaf() * 0.1).astype(np.uint8))
    assert dark["status"] in ("low_quality", "no_leaf")
    blurred = vision.quality_check(cv2.GaussianBlur(synth_leaf(spots=5), (0, 0), 25))
    assert blurred["status"] == "low_quality", blurred["status"]
    grey = np.full((256, 256, 3), BG, np.uint8)
    assert vision.quality_check(grey)["status"] == "no_leaf"
    two = np.full((256, 512, 3), BG, np.uint8)
    cv2.ellipse(two, (128, 128), (60, 90), 0, 0, 360, (45, 150, 55), -1)
    cv2.ellipse(two, (384, 128), (60, 90), 0, 0, 360, (45, 150, 55), -1)
    assert vision.quality_check(two)["status"] == "multiple_leaves"


def test_prepare_leaf_crops_offcentre_leaf():
    big = np.full((600, 900, 3), BG, np.uint8)
    cv2.ellipse(big, (700, 200), (80, 120), 0, 0, 360, (45, 150, 55), -1)
    out = vision.prepare_leaf(big)
    assert out.shape == (384, 384, 3)
    assert vision.leaf_mask(out)[0].mean() > 0.3          # leaf now fills a decent part of the frame


def test_risk_direction():
    wet = risk.assess_risk("early_blight", 20, 27, 90, 15, True)["score"]
    dry = risk.assess_risk("early_blight", 20, 12, 30, 0, False)["score"]
    assert wet > dry + 30
    mites_dry = risk.assess_risk("spider_mites", 20, 32, 25, 0, False)["score"]
    mites_wet = risk.assess_risk("spider_mites", 20, 20, 90, 20, True)["score"]
    assert mites_dry > mites_wet                           # mites like it dry
    r = risk.assess_risk("late_blight", 10, None, None, None, False)
    assert not r["env_complete"] and 0 <= r["score"] <= 100
    assert abs(sum(f["contribution"] for f in r["factors"]) - r["score"]) <= 1


def test_forecast_properties():
    lo = risk.forecast("early_blight", 23, 20)
    hi = risk.forecast("early_blight", 23, 90)
    for f in (lo, hi):
        c = f["curve"]
        assert c[0]["median"] == 23.0 and c[0]["low"] == c[0]["high"] == 23.0
        assert all(c[i]["median"] <= c[i + 1]["median"] + 1e-9 for i in range(14))      # monotone
        assert all(0 <= p["low"] <= p["median"] <= p["high"] <= 100 for p in c)
    assert hi["milestones"]["day_7"]["median"] > lo["milestones"]["day_7"]["median"]
    print("day7 median low-risk/high-risk:", lo["milestones"]["day_7"]["median"], hi["milestones"]["day_7"]["median"])
    assert risk.forecast("early_blight", 23, 50, seed=0) == risk.forecast("early_blight", 23, 50, seed=0)  # reproducible


def test_forecast_uses_history():
    hist = [(-8, 10), (-4, 22), (-1, 35), (0, 40)]
    f = risk.forecast("early_blight", 40, 50, hist)
    assert "history" in f["source"]


def test_forecast_evaluation():
    c = [30 + i for i in range(15)]                         # predicts +1 %/day from 30
    rows = [{"id": 1, "t": 0, "severity": 30, "curve": c},
            {"id": 2, "t": 3 * 86400, "severity": 35, "curve": c},
            {"id": 3, "t": 7 * 86400, "severity": 40, "curve": None}]
    ev = risk.evaluate_forecasts(rows)
    assert ev["n"] == 2 and ev["mae"] is not None and ev["r2"] is None   # r2 needs >=3 pairs
    assert risk.evaluate_forecasts([rows[0]])["mae"] is None


def test_recommendation_priority():
    r = risk.assess_risk("early_blight", 40, 27, 90, 15, True)
    p = risk.forecast("early_blight", 40, r["score"])
    hi = rec_mod.recommend("Tomato", "Early blight", "early_blight", 40, "High", r, p)
    assert hi["priority"] == "HIGH" and "Rule" in hi["evidence"]
    r2 = risk.assess_risk("early_blight", 2, 10, 30, 0, False)
    p2 = risk.forecast("early_blight", 2, r2["score"])
    lo = rec_mod.recommend("Tomato", "Early blight", "early_blight", 2, "Low", r2, p2)
    assert lo["priority"] == "LOW"
    assert rec_mod.recommend("Tomato", "Healthy", None, 0, "None", None, None)["priority"] == "LOW"


def test_db_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        db.init(path)
        pid = db.create_plant("Tomato #1", "Tomato", None, path)
        aid = db.save_analysis(pid, time.time(), dict(class_name="x", crop="Tomato", disease="Early blight", disease_key="early_blight",
                               confidence=90.0, severity_pct=12.0, severity_level="Moderate", curve=[{"median": 12.0}] * 15), path)
        db.update_report(aid, {"hello": 1}, path)
        assert db.get_analysis(aid, path) == {"hello": 1}
        h = db.history(pid, path)
        assert len(h) == 1 and h[0]["curve"] == [12.0] * 15 and db.list_plants(path)[0]["scans"] == 1


class FakeService:
    """Stands in for ModelService (no TensorFlow)."""
    class_names = ["Tomato_Early_blight", "Tomato_Late_blight", "Tomato_healthy"]
    low_conf, version = 0.6, "fake-test"
    def __init__(self, p): self.p = np.array(p)
    def predict(self, rgb):
        cam = np.zeros((7, 7), np.float32); cam[2:5, 2:5] = 1
        return self.p, cam


def test_pipeline_ok_lowconf_healthy_and_failures():
    img = enc(synth_leaf(spots=5))
    rep, ims = run_analysis(FakeService([0.9, 0.07, 0.03]), img, {"temperature": 27, "humidity": 85, "rainfall_mm": 10, "recent_rain": True, "observed_at": time.time()})
    assert rep["status"] == "ok" and rep["disease"]["name"] == "Early blight" and rep["crop"]["name"] == "Tomato"
    assert rep["severity"]["affected_percent"] > 0 and rep["risk"]["score"] > 0 and rep["progression"]["milestones"]["day_7"]
    assert set(ims) == {"original", "heatmap", "cam_overlay", "mask", "seg_overlay"}
    low, _ = run_analysis(FakeService([0.4, 0.35, 0.25]), img, {})
    assert low["status"] == "low_confidence" and "disease" not in low
    hl, _ = run_analysis(FakeService([0.02, 0.03, 0.95]), img, {})
    assert hl["disease"]["is_healthy"] and hl["severity"]["applicable"] is False and hl["risk"] is None
    bad, _ = run_analysis(FakeService([1, 0, 0]), b"not an image", {})
    assert bad["status"] == "invalid_image"
    nl, _ = run_analysis(FakeService([1, 0, 0]), enc(np.full((200, 200, 3), BG, np.uint8)), {})
    assert nl["status"] == "no_leaf" and "disease" not in nl


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"\n{len(fns)} tests passed")
