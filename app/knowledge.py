"""Static domain knowledge used by the risk, progression and recommendation modules.

HONESTY NOTE
------------
Everything in RISK_PROFILES and GROWTH_RATES is an *expert assumption* based on general
plant-pathology knowledge (favourable temperature/humidity ranges, typical speed of lesion
expansion). None of it is fitted to data. Before submitting the project, attach a citation
(extension bulletin, paper) to every row in docs/ and adjust the numbers if the source differs.
The UI labels these modules "rule-based model" / "projection" accordingly.
"""
from __future__ import annotations

# class folder name (as produced by the training notebook) -> (crop, disease label, disease key)
# key None = healthy class
CLASS_INFO = {
    "Pepper__bell___Bacterial_spot": ("Pepper", "Bacterial spot", "bacterial_spot"),
    "Pepper__bell___healthy": ("Pepper", "Healthy", None),
    "Potato___Early_blight": ("Potato", "Early blight", "early_blight"),
    "Potato___Late_blight": ("Potato", "Late blight", "late_blight"),
    "Potato___healthy": ("Potato", "Healthy", None),
    "Tomato_Bacterial_spot": ("Tomato", "Bacterial spot", "bacterial_spot"),
    "Tomato_Early_blight": ("Tomato", "Early blight", "early_blight"),
    "Tomato_Late_blight": ("Tomato", "Late blight", "late_blight"),
    "Tomato_Leaf_Mold": ("Tomato", "Leaf mold", "leaf_mold"),
    "Tomato_Septoria_leaf_spot": ("Tomato", "Septoria leaf spot", "septoria"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider mites", "spider_mites"),
    "Tomato__Target_Spot": ("Tomato", "Target spot", "target_spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("Tomato", "Yellow leaf curl virus", "tylcv"),
    "Tomato__Tomato_mosaic_virus": ("Tomato", "Mosaic virus", "tomv"),
    "Tomato_healthy": ("Tomato", "Healthy", None),
}


def parse_class(name: str):
    """Return (crop, disease_label, key). Falls back gracefully for unknown folder names."""
    if name in CLASS_INFO:
        return CLASS_INFO[name]
    crop = name.split("_")[0].title() or "Unknown"
    rest = name[len(name.split("_")[0]):].strip("_").replace("_", " ").strip()
    if "healthy" in name.lower():
        return crop, "Healthy", None
    return crop, rest or "Unknown", "generic"


# --- Environmental risk rule model ------------------------------------------------------
# temp = trapezoid (min, optimum_low, optimum_high, max) in deg C -> 0..1 favourability
# hum  = (low, high) relative humidity % ramp; inverse=True means DRY conditions favour the problem
# rain_ref = mm of rain at which the rain factor saturates
# w = weights (temperature, humidity, rain, current severity); must sum to 1
RISK_PROFILES = {
    "early_blight":   dict(temp=(10, 24, 29, 35), hum=(60, 90), inverse=False, rain_ref=10, w=(0.30, 0.30, 0.20, 0.20)),
    "late_blight":    dict(temp=(5, 12, 20, 28),  hum=(75, 95), inverse=False, rain_ref=8,  w=(0.30, 0.35, 0.15, 0.20)),
    "bacterial_spot": dict(temp=(15, 24, 30, 38), hum=(70, 95), inverse=False, rain_ref=10, w=(0.25, 0.25, 0.30, 0.20)),
    "leaf_mold":      dict(temp=(12, 20, 25, 32), hum=(80, 95), inverse=False, rain_ref=5,  w=(0.25, 0.45, 0.10, 0.20)),
    "septoria":       dict(temp=(10, 20, 25, 32), hum=(75, 95), inverse=False, rain_ref=10, w=(0.25, 0.30, 0.25, 0.20)),
    "target_spot":    dict(temp=(15, 22, 30, 36), hum=(70, 95), inverse=False, rain_ref=8,  w=(0.30, 0.30, 0.20, 0.20)),
    "spider_mites":   dict(temp=(18, 27, 35, 42), hum=(20, 60), inverse=True,  rain_ref=10, w=(0.35, 0.25, 0.20, 0.20)),
    # vector / mechanically transmitted viruses: weather is a weak, indirect driver
    "tylcv":          dict(temp=(18, 25, 32, 40), hum=(30, 70), inverse=True,  rain_ref=15, w=(0.40, 0.05, 0.10, 0.45)),
    "tomv":           dict(temp=(15, 22, 30, 38), hum=(30, 70), inverse=False, rain_ref=15, w=(0.30, 0.05, 0.05, 0.60)),
    "generic":        dict(temp=(10, 20, 30, 38), hum=(60, 90), inverse=False, rain_ref=10, w=(0.30, 0.30, 0.20, 0.20)),
}

RISK_NOTES = {
    "tylcv": "Weather acts on the whitefly vector, not on the virus directly - treat the weather part of this score as weak evidence.",
    "tomv": "This virus spreads mechanically (hands, tools, seed); weather is only a weak driver, so the score is dominated by current severity.",
    "spider_mites": "Mites are favoured by hot, DRY conditions, so low humidity and no rain raise the score.",
}

# --- Progression: logistic growth rate range (per day) at risk = 0 and risk = 100 --------
# ASSUMED plausible ranges, not fitted to data (see honesty note).
GROWTH_RATES = {
    "early_blight": (0.05, 0.25), "late_blight": (0.08, 0.40), "bacterial_spot": (0.05, 0.22),
    "leaf_mold": (0.04, 0.18), "septoria": (0.05, 0.22), "target_spot": (0.04, 0.18),
    "spider_mites": (0.05, 0.25), "tylcv": (0.03, 0.12), "tomv": (0.02, 0.08), "generic": (0.04, 0.18),
}

# --- Recommendations: cultural / sanitation measures only (no products, no doses) ---------
ACTIONS = {
    "early_blight": [
        "Remove and bag heavily spotted lower leaves; do not compost them.",
        "Water at the base in the morning and avoid wetting the foliage.",
        "Improve airflow: adequate spacing, staking and pruning of lower leaves.",
        "Mulch to reduce soil splash onto leaves.",
        "Rotate away from tomato/potato-family crops next season.",
    ],
    "late_blight": [
        "Inspect neighbouring plants daily - late blight can spread very quickly in cool, wet weather.",
        "Remove and bag infected leaves/plants; do not compost them.",
        "Avoid overhead irrigation and keep foliage dry.",
        "For potato, do not save tubers from infected plants for seed.",
        "Contact your local agricultural extension service promptly for approved management options.",
    ],
    "bacterial_spot": [
        "Avoid handling plants while they are wet to limit spread.",
        "Remove badly infected leaves and disinfect pruning tools.",
        "Use drip irrigation instead of overhead watering.",
        "Use clean seed/transplants and rotate crops.",
    ],
    "leaf_mold": [
        "Lower humidity: ventilate greenhouses/tunnels and increase plant spacing.",
        "Avoid wetting leaves; water early in the day.",
        "Remove infected leaves and clear plant debris.",
        "Consider resistant varieties next season.",
    ],
    "septoria": [
        "Remove infected lower leaves and clear plant debris.",
        "Mulch to prevent soil splash; avoid overhead watering.",
        "Rotate crops and keep plants well spaced.",
    ],
    "target_spot": [
        "Improve air circulation by pruning and spacing.",
        "Remove infected leaves; avoid overhead watering.",
        "Clear crop debris at the end of the season and rotate crops.",
    ],
    "spider_mites": [
        "Check the undersides of leaves for mites and fine webbing.",
        "A strong water spray can reduce numbers; reduce dust and water stress.",
        "Protect natural predators; avoid broad-spectrum insecticides that harm them.",
        "Ask local extension about approved miticides if the infestation grows.",
    ],
    "tylcv": [
        "There is no cure for infected plants - remove and bag severely affected plants.",
        "Control whiteflies: yellow sticky traps, insect-proof netting, reflective mulch.",
        "Check nearby plants for whiteflies and early yellowing/curling.",
        "Use resistant/tolerant varieties in future plantings.",
    ],
    "tomv": [
        "There is no cure - remove and dispose of infected plants.",
        "Wash hands and disinfect tools between plants.",
        "Avoid handling tobacco products before touching plants.",
        "Use clean seed and resistant varieties.",
    ],
    "generic": [
        "Isolate and monitor the affected plant.",
        "Remove badly affected leaves and keep foliage dry.",
        "Consult local agricultural extension for a confirmed diagnosis.",
    ],
    "healthy": [
        "No disease detected - keep monitoring regularly.",
        "Keep watering consistent and avoid wetting leaves.",
        "Maintain good airflow and remove fallen debris.",
    ],
}

DISCLAIMER = (
    "Decision support only - not a laboratory diagnosis. Follow locally approved agricultural "
    "guidance for any chemical treatment (product, dose and timing)."
)
