"""Classical computer-vision helpers: quality gate, leaf cropping, lesion segmentation, overlays.

The leaf/lesion segmentation is a COLOUR-BASED HEURISTIC, not a trained segmenter. It works best
on a single leaf against a plain background. Its accuracy has not been measured until you
hand-label some images and compute IoU/Dice (see README -> Limitations).
"""
from __future__ import annotations
import io
import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1600
BLUR_FAIL, BLUR_WARN = 15.0, 40.0        # variance of Laplacian (tune on your own photos)
DARK_FAIL, BRIGHT_FAIL = 35.0, 235.0     # mean gray level
MIN_LEAF_FRAC = 0.08                     # leaf must cover at least 8 % of the frame
SEVERITY_BANDS = ((10.0, "Low"), (30.0, "Moderate"), (60.0, "High"))  # >60 -> Critical


def severity_level(pct: float) -> str:
    for upper, name in SEVERITY_BANDS:
        if pct <= upper:
            return name
    return "Critical"


def decode_image(data: bytes):
    """bytes -> BGR ndarray (EXIF-rotated, capped at MAX_SIDE) or None."""
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _limit(bgr, side):
    h, w = bgr.shape[:2]
    if max(h, w) <= side:
        return bgr
    s = side / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def leaf_mask(bgr):
    """Vegetation-coloured region (green/yellow/brown hues), holes filled.
    Returns (mask 0/1, contours sorted by area desc, areas)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    veg = ((h >= 8) & (h <= 100) & (s >= 35) & (v >= 25)).astype(np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(veg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = veg.size
    cnts = sorted((c for c in cnts if cv2.contourArea(c) >= 0.02 * total), key=cv2.contourArea, reverse=True)
    filled = np.zeros_like(veg)
    cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
    return filled, cnts, [cv2.contourArea(c) for c in cnts]


def quality_check(bgr):
    """Gate run BEFORE the classifier. status: ok | no_leaf | multiple_leaves | low_quality."""
    small = _limit(bgr, 512)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    mask, _, areas = leaf_mask(small)
    frac = float(mask.mean())
    total = mask.size
    multi = len([a for a in areas if a >= 0.10 * total and a >= 0.4 * areas[0]]) >= 2 if areas else False

    has_leaf = frac >= MIN_LEAF_FRAC
    sharp_ok, light_ok = blur >= BLUR_FAIL, DARK_FAIL <= bright <= BRIGHT_FAIL
    checks = [
        {"name": "Leaf detected", "pass": has_leaf, "detail": f"{frac * 100:.0f}% of the frame looks like leaf tissue"},
        {"name": "Single leaf", "pass": not multi, "detail": "more than one leaf-sized region found" if multi else "one main leaf region"},
        {"name": "Image clarity", "pass": sharp_ok, "warn": sharp_ok and blur < BLUR_WARN, "detail": f"sharpness index {blur:.0f}"},
        {"name": "Lighting", "pass": light_ok, "detail": f"mean brightness {bright:.0f}/255"},
    ]
    blur_s = float(np.clip((blur - BLUR_FAIL) / (100 - BLUR_FAIL), 0, 1))
    light_s = float(1 - np.clip(abs(bright - 130) / 110, 0, 1))
    leaf_s = float(np.clip(frac / 0.30, 0, 1))
    score = round(100 * (0.4 * blur_s + 0.3 * light_s + 0.3 * leaf_s))

    tips, status = [], "ok"
    if not has_leaf:
        status, tips = "no_leaf", ["Place one leaf in the guide box", "Use a plain background", "Move closer"]
    elif multi:
        status, tips = "multiple_leaves", ["Capture one clear leaf", "Separate the leaf from others"]
    elif not (sharp_ok and light_ok):
        status = "low_quality"
        if not sharp_ok:
            tips += ["Move closer", "Keep the camera steady", "Tap to focus and retake"]
        if bright < DARK_FAIL:
            tips.append("Improve lighting (the image is too dark)")
        if bright > BRIGHT_FAIL:
            tips.append("Reduce glare (the image is too bright)")
    return {"status": status, "score": score, "checks": checks, "tips": tips,
            "metrics": {"sharpness": round(blur, 1), "brightness": round(bright, 1), "leaf_fraction": round(frac, 3)}}


def prepare_leaf(bgr, out=384):
    """Crop to the main leaf (square, small margin) and resize to out x out (BGR)."""
    h, w = bgr.shape[:2]
    work = _limit(bgr, 512)
    sc = work.shape[1] / w
    _, cnts, _ = leaf_mask(work)
    crop = None
    if cnts:
        x, y, bw, bh = [int(round(t / sc)) for t in cv2.boundingRect(cnts[0])]
        side = max(bw, bh, 8)
        if not (abs(w - h) <= 2 and side >= 0.6 * max(w, h)):   # square leaf-filled images stay untouched
            side = int(side * 1.16)
            cx, cy = x + bw // 2, y + bh // 2
            pad = side // 2 + 2
            padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0 = cx - side // 2 + pad, cy - side // 2 + pad
            crop = padded[y0:y0 + side, x0:x0 + side]
    if crop is None:                                            # centre square fallback
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = bgr[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def segment(bgr):
    """Leaf + lesion masks and severity = lesion pixels / leaf pixels x 100."""
    leaf, _, _ = leaf_mask(bgr)
    leaf = cv2.erode(leaf, np.ones((3, 3), np.uint8), iterations=2)   # drop blended edge pixels
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    healthy = (h >= 36) & (h <= 95) & (s >= 40) & (v >= 35)           # "normal green" tissue
    les = ((leaf > 0) & ~healthy).astype(np.uint8)
    les = cv2.morphologyEx(les, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    les = (cv2.medianBlur(les * 255, 5) // 255).astype(np.uint8) & leaf
    leaf_px, les_px = int(leaf.sum()), int(les.sum())
    pct = 100.0 * les_px / leaf_px if leaf_px else 0.0
    if pct < 0.5:                                                      # ignore speckle noise
        les[:] = 0
        les_px, pct = 0, 0.0
    return {"leaf": leaf, "lesion": les, "leaf_pixels": leaf_px, "lesion_pixels": les_px, "percent": pct}


def seg_images(bgr, leaf, les):
    """(mask picture, overlay picture): green = healthy tissue, red = affected."""
    healthy = (leaf > 0) & (les == 0)
    mask = np.zeros_like(bgr)
    mask[healthy] = (0, 190, 0)
    mask[les > 0] = (0, 0, 255)
    ov = bgr.astype(np.float32)
    ov[healthy] = ov[healthy] * 0.6 + np.array([0, 200, 0], np.float32) * 0.4
    ov[les > 0] = ov[les > 0] * 0.35 + np.array([0, 0, 255], np.float32) * 0.65
    return mask, ov.astype(np.uint8)


def heat_images(bgr, cam):
    """(pure heatmap, overlay) from a Grad-CAM map in [0,1] of any size."""
    cam = cv2.resize(cam.astype(np.float32), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colored, cv2.addWeighted(bgr, 0.55, colored, 0.45, 0)


def jpg_bytes(img, q=88):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes()
