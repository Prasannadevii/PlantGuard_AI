"""Environmental risk (rule-based) and disease progression (logistic projection).

Neither module is a trained ML model and the API/UI say so explicitly:
  * risk        -> transparent weighted rule model (parameters in knowledge.RISK_PROFILES)
  * progression -> logistic growth scenario; Monte-Carlo spread over assumed parameter ranges.
                   The band is SCENARIO SPREAD, not a statistical prediction interval.
                   When >=3 scans of the same plant exist, the growth rate is partly fitted
                   to that plant's own history.
"""
from __future__ import annotations
import numpy as np
from .knowledge import RISK_PROFILES, RISK_NOTES, GROWTH_RATES

RISK_METHOD = "Rule-based weighted model (expert assumptions, not trained ML)"
HORIZON = 14


def _trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)


def risk_level(score: float) -> str:
    return "Low" if score < 35 else "Moderate" if score <= 65 else "High"


def assess_risk(key, severity_pct, temperature=None, humidity=None, rainfall=None, recent_rain=False):
    """Return {score, level, method, factors[], note, env_complete}. Missing inputs count as neutral (0.5)."""
    p = RISK_PROFILES.get(key, RISK_PROFILES["generic"])
    wt, wh, wr, ws = p["w"]
    inv = p["inverse"]
    provided = {"temperature": temperature is not None, "humidity": humidity is not None,
                "rainfall": rainfall is not None or recent_rain}

    f_t = _trap(temperature, *p["temp"]) if temperature is not None else 0.5
    if humidity is not None:
        lo, hi = p["hum"]
        f_h = float(np.clip((humidity - lo) / (hi - lo), 0, 1))
        f_h = 1 - f_h if inv else f_h
    else:
        f_h = 0.5
    if rainfall is not None or recent_rain:
        f_r = float(np.clip((rainfall or 0.0) / p["rain_ref"], 0, 1))
        if recent_rain:
            f_r = max(f_r, 0.6)
        f_r = 1 - f_r if inv else f_r
    else:
        f_r = 0.5
    f_s = float(np.clip(severity_pct / 60.0, 0, 1))

    rows = [
        ("Temperature", f"{temperature:g} °C" if temperature is not None else "not provided", f_t, wt),
        ("Humidity", f"{humidity:g} %" if humidity is not None else "not provided", f_h, wh),
        ("Rainfall", (f"{rainfall or 0:g} mm" + (", recent rain" if recent_rain else "")) if provided["rainfall"] else "not provided", f_r, wr),
        ("Current severity", f"{severity_pct:.1f} %", f_s, ws),
    ]
    factors = [{"name": n, "value": v, "favourability": round(100 * f), "weight": w,
                "contribution": round(100 * f * w, 1)} for n, v, f, w in rows]
    score = round(sum(f["contribution"] for f in factors))
    return {"score": score, "level": risk_level(score), "method": RISK_METHOD, "factors": factors,
            "note": RISK_NOTES.get(key), "env_complete": all(provided.values())}


def _fit_rate(points, K=100.0):
    """Least-squares slope of logit(S/K) vs time. points = [(t_days, severity_pct)]. -> (slope, se) or None."""
    if len(points) < 3:
        return None
    t = np.array([p[0] for p in points], float)
    s = np.clip(np.array([p[1] for p in points], float), 1.0, 99.0)
    if t.max() - t.min() < 2:
        return None
    y = np.log(s / (K - s))
    A = np.vstack([t, np.ones_like(t)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    se = float(np.sqrt((resid @ resid) / (len(t) - 2) / ((t - t.mean()) ** 2).sum()))
    return float(coef[0]), se


def forecast(key, severity_pct, risk_score=None, history=None, n=2000, seed=0):
    """Logistic projection of lesion-area severity for days 0..14.
    history = [(t_days_relative_to_now, severity_pct), ...] for the SAME plant and disease (incl. current)."""
    lo, hi = GROWTH_RATES.get(key, GROWTH_RATES["generic"])
    risk = 50.0 if risk_score is None else float(risk_score)
    r_mid = lo + (hi - lo) * risk / 100.0
    rel = 0.40 if risk_score is not None else 0.55          # wider spread if environment unknown
    source, r_center = "prior (assumed growth range scaled by environmental risk)", r_mid
    fit = _fit_rate(history or [])
    if fit:
        slope, se = fit
        w = len(history) / (len(history) + 3.0)
        r_center = w * max(slope, 0.0) + (1 - w) * r_mid
        rel = float(np.clip((w * se + (1 - w) * rel * r_mid) / max(r_center, 1e-3), 0.15, 0.6))
        source = f"prior blended with this plant's own history ({len(history)} scans, weight {w:.2f})"

    s0 = float(np.clip(severity_pct, 1.0, 99.0))            # 1 % floor so growth can start
    rng = np.random.default_rng(seed)
    r = r_center * np.exp(rng.normal(0, rel, n))
    K = np.maximum(rng.uniform(70, 100, n), s0 + 1)
    t = np.arange(HORIZON + 1)[:, None]
    S = K / (1 + ((K - s0) / s0) * np.exp(-r * t))
    p10, p50, p90 = np.percentile(S, [10, 50, 90], axis=1)
    curve = [{"day": int(d), "median": round(float(p50[d]), 1), "low": round(float(p10[d]), 1), "high": round(float(p90[d]), 1)}
             for d in range(HORIZON + 1)]
    return {
        "method": "Logistic growth projection with Monte-Carlo parameter spread (not a trained forecaster)",
        "source": source,
        "growth_rate_per_day": round(float(r_center), 3),
        "curve": curve,
        "milestones": {f"day_{d}": curve[d] for d in (3, 7, 14)},
        "severity_floor_applied": severity_pct < 1.0,
        "uncertainty_note": "Band = 10th-90th percentile over assumed growth-rate/ceiling ranges (scenario spread), "
                            "not a statistical prediction interval. Assumes conditions stay similar and no treatment.",
    }


def evaluate_forecasts(rows):
    """Pair each stored forecast with the NEXT later scan (0.5-14 days after) and score it.
    rows: [{id, t (epoch s), severity, curve [15 medians] | None}] sorted by t."""
    pairs = []
    for i, a in enumerate(rows):
        if not a.get("curve"):
            continue
        for b in rows[i + 1:]:
            d = (b["t"] - a["t"]) / 86400.0
            if d < 0.5:
                continue
            if d <= HORIZON:
                pred = float(np.interp(d, np.arange(HORIZON + 1), a["curve"]))
                pairs.append({"from_id": a["id"], "to_id": b["id"], "days": round(d, 1),
                              "predicted": round(pred, 1), "actual": round(b["severity"], 1)})
            break
    out = {"pairs": pairs, "n": len(pairs), "mae": None, "rmse": None, "r2": None}
    if pairs:
        e = np.array([p["predicted"] - p["actual"] for p in pairs])
        out["mae"], out["rmse"] = round(float(np.abs(e).mean()), 2), round(float(np.sqrt((e ** 2).mean())), 2)
        act = np.array([p["actual"] for p in pairs])
        if len(pairs) >= 3 and act.var() > 0:
            out["r2"] = round(float(1 - (e ** 2).sum() / ((act - act.mean()) ** 2).sum()), 3)
    return out
