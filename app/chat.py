"""PlantGuard AI Assistant: Groq LLM that only EXPLAINS the structured results produced by the ML pipeline."""
from __future__ import annotations
import json
import requests
from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_URL

SYSTEM = (
    "You are the PlantGuard AI Assistant. A computer-vision pipeline has already analysed a leaf; its structured "
    "results are given below as JSON. Explain them in plain, friendly language.\n"
    "Rules: use ONLY numbers and facts present in the JSON; never invent probabilities, percentages, diagnoses, "
    "pesticide names or doses; if something is not in the JSON, say you do not have it. Remember that the risk score "
    "comes from a rule-based model and the progression is a scenario projection, not a certain forecast, and Grad-CAM "
    "shows what influenced the model, not biological proof. For chemical treatment, tell the user to consult local "
    "agricultural extension and follow product labels. Keep answers under 180 words."
)


def compact_context(r: dict) -> dict:
    """Strip images/URLs; keep only the numbers and labels the assistant may talk about."""
    g = r.get
    return {
        "crop": g("crop", {}).get("name"), "crop_confidence_pct": g("crop", {}).get("confidence"),
        "disease": g("disease", {}).get("name"), "disease_confidence_pct": g("disease", {}).get("confidence"),
        "top3": g("top3"), "image_quality": {"score": g("quality", {}).get("score")},
        "severity": {k: g("severity", {}).get(k) for k in ("applicable", "affected_percent", "healthy_percent", "level")},
        "environment": g("environment"),
        "risk": {k: (g("risk") or {}).get(k) for k in ("score", "level", "method", "factors")} if g("risk") else None,
        "progression": {"source": g("progression", {}).get("source"), "milestones": g("progression", {}).get("milestones"),
                        "note": g("progression", {}).get("uncertainty_note")} if g("progression") else None,
        "recommendation": {k: g("recommendation", {}).get(k) for k in ("priority", "actions", "next_scan_days", "evidence")},
    }


def ask(report: dict, message: str, history: list | None = None) -> dict:
    if not GROQ_API_KEY:
        return {"configured": False, "reply": "The assistant is not configured yet. Add GROQ_API_KEY to the .env file "
                                              "and restart the server."}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "system", "content": "ANALYSIS_JSON: " + json.dumps(compact_context(report))}]
    for m in (history or [])[-6:]:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"], "content": m["content"][:1500]})
    msgs.append({"role": "user", "content": message[:1000]})
    try:
        r = requests.post(GROQ_URL, timeout=30,
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                          json={"model": GROQ_MODEL, "messages": msgs, "temperature": 0.2, "max_tokens": 450})
        if r.status_code != 200:
            return {"configured": True, "reply": f"The assistant is unavailable (Groq returned {r.status_code}). "
                                                 "Check GROQ_API_KEY and GROQ_MODEL in .env."}
        return {"configured": True, "reply": r.json()["choices"][0]["message"]["content"].strip()}
    except Exception as e:  # network errors, bad JSON ...
        return {"configured": True, "reply": f"The assistant could not be reached ({type(e).__name__})."}
