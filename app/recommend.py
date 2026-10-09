"""Rule-based recommendation engine: structured inputs in, traceable advice out."""
from __future__ import annotations
from .knowledge import ACTIONS, DISCLAIMER

_SEV_POINTS = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
_RISK_POINTS = {"Low": 0, "Moderate": 1, "High": 2}


def recommend(crop, disease, key, severity_pct, severity_level, risk, progression):
    if key is None:
        return {"priority": "LOW", "actions": list(ACTIONS["healthy"]), "next_scan_days": "7",
                "evidence": {"Crop": crop, "Finding": "Healthy leaf", "Rule": "no disease detected -> preventive monitoring"},
                "disclaimer": DISCLAIMER}

    p7 = progression["milestones"]["day_7"]["median"] if progression else severity_pct
    prog_pts = 0 if p7 < 20 else 1 if p7 < 45 else 2
    pts = _SEV_POINTS[severity_level] + (_RISK_POINTS[risk["level"]] if risk else 1) + prog_pts
    priority = "LOW" if pts <= 2 else "MEDIUM" if pts <= 4 else "HIGH"

    actions = list(ACTIONS.get(key, ACTIONS["generic"]))
    if severity_level in ("High", "Critical"):
        actions.insert(0, "Severity is high: remove or isolate the most affected plant parts today.")
    if priority == "HIGH":
        actions.append("Check neighbouring plants now and consider asking local agricultural extension for advice.")
    nxt = {"HIGH": "2-3", "MEDIUM": "3-5", "LOW": "5-7"}[priority]

    evidence = {
        "Crop": crop, "Disease": disease,
        "Severity": f"{severity_pct:.1f}% ({severity_level})",
        "Environmental risk": f"{risk['score']}/100 ({risk['level']})" if risk else "not computed",
        "Projected severity (day 7, median)": f"{p7:.0f}%",
        "Rule": f"priority points = severity {_SEV_POINTS[severity_level]} + risk "
                f"{_RISK_POINTS[risk['level']] if risk else 1} + progression {prog_pts} = {pts} -> {priority}",
    }
    return {"priority": priority, "actions": actions, "next_scan_days": nxt, "evidence": evidence, "disclaimer": DISCLAIMER}
