# Methods and assumptions

## Classification
Transfer learning (ImageNet weights) with augmentation inside the model. Candidates: baseline CNN, MobileNetV2, EfficientNetB0.
Selection on **validation macro-F1**; top backbone layers fine-tuned only if validation macro-F1 improves; the **test set is evaluated once**.
Split: 70/15/15, stratified, with near-duplicate images grouped (16×16 grayscale fingerprints, correlation over 8 rotations/flips,
threshold auto-chosen so no group exceeds 40 images) so rotated copies of one leaf cannot leak across splits.
Class imbalance: square-root-dampened balanced class weights. Calibration: temperature scaling on the validation set.
Reported: accuracy, macro precision/recall/F1, per-class table, confusion matrix, train/val curves, ECE, coverage-vs-accuracy at confidence thresholds.

## Severity
`severity % = lesion pixels / leaf pixels × 100`.
Leaf = vegetation-coloured region (HSV hue 8–100, S≥35, V≥25), holes filled, eroded 2 px to drop blended edges.
"Healthy tissue" = HSV hue 36–95, S≥40, V≥35; any other leaf pixel counts as affected; speckle removed by opening + median filter; <0.5 % is treated as 0.
Bands: ≤10 Low · ≤30 Moderate · ≤60 High · >60 Critical (illustrative; edit `SEVERITY_BANDS`).
Healthy predictions skip lesion measurement (severity shown as n/a).

## Environmental risk (rule-based, not ML)
`score = 100 × Σ weight_i × favourability_i` over temperature, humidity, rainfall and current severity.
Temperature = trapezoid (min, optimum range, max); humidity = linear ramp between two RH values (inverted for spider mites);
rainfall = ramp up to a reference amount (recent rain ⇒ at least 0.6; inverted for mites). Missing inputs count as 0.5 (neutral) and are flagged.
Levels: <35 Low, 35–65 Moderate, >65 High.

| Disease key | Temp trapezoid °C | Humidity ramp % | Rain ref mm | Weights (T,H,R,Sev) | Source |
|---|---|---|---|---|---|
| early_blight | 10/24/29/35 | 60–90 | 10 | .30/.30/.20/.20 | TODO cite |
| late_blight | 5/12/20/28 | 75–95 | 8 | .30/.35/.15/.20 | TODO cite |
| bacterial_spot | 15/24/30/38 | 70–95 | 10 | .25/.25/.30/.20 | TODO cite |
| leaf_mold | 12/20/25/32 | 80–95 | 5 | .25/.45/.10/.20 | TODO cite |
| septoria | 10/20/25/32 | 75–95 | 10 | .25/.30/.25/.20 | TODO cite |
| target_spot | 15/22/30/36 | 70–95 | 8 | .30/.30/.20/.20 | TODO cite |
| spider_mites | 18/27/35/42 | 20–60 (dry favours) | 10 (inverted) | .35/.25/.20/.20 | TODO cite |
| tylcv | 18/25/32/40 | 30–70 (inverted) | 15 (inverted) | .40/.05/.10/.45 | TODO cite |
| tomv | 15/22/30/38 | 30–70 | 15 | .30/.05/.05/.60 | TODO cite |

These values are general plant-pathology assumptions that have **not** been checked against specific sources: fill the *Source* column
(university extension bulletins, review papers) and correct any number the source contradicts.

## Progression (projection, not a validated forecast)
Logistic growth `S(t) = K / (1 + ((K − S0)/S0) · e^(−r·t))`, S0 = measured severity (floored at 1 %), K ~ U(70, 100) %.
`r = r_lo + (r_hi − r_lo) × risk/100` using assumed per-disease ranges (`GROWTH_RATES`), then 2000 Monte-Carlo draws with log-normal noise on `r`.
The band is the 10th–90th percentile of those draws - **scenario spread, not a statistical prediction interval**.
With ≥3 scans of the same plant and disease, `r` is blended with the slope of logit(S/100) vs time fitted to that history (weight n/(n+3)).
Assumes unchanged conditions and no treatment.

## Prediction vs actual
Each stored forecast curve is compared with the **next** scan taken 0.5–14 days later (linear interpolation of the stored median curve).
MAE and RMSE need ≥1 pair; R² needs ≥3 pairs with varying actual values. Photos taken under different conditions add measurement noise to "actual".

## Recommendation priority
points = severity (Low 0, Moderate 1, High 2, Critical 3) + risk (Low 0, Moderate 1, High 2) + projected day-7 severity (<20 % 0, <45 % 1, else 2);
≤2 LOW, 3–4 MEDIUM, ≥5 HIGH. Actions are cultural/sanitation measures per disease; no pesticide products or doses are given.
