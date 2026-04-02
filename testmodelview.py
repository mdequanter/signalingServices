import cv2
import numpy as np
import time
from ultralytics import YOLO

# =========================
# INSTELLINGEN
# =========================
MODEL_PATH = "models\\denham.pt"
IMAGE_PATH = "path2.jpg"
CONF = 0.25
ALPHA = 0.4
SHOW_LABELS = True

# =========================
# MODEL LADEN
# =========================
model = YOLO(MODEL_PATH)

# =========================
# AFBEELDING LADEN
# =========================
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Kon afbeelding niet laden: {IMAGE_PATH}")

# =========================
# INFERENTIE (timing meten)
# =========================
start_time = time.time()

results = model.predict(
    source=image,
    conf=CONF,
    save=False,
    verbose=False,
    retina_masks=True
)

end_time = time.time()
total_time = (end_time - start_time) * 1000  # ms

# =========================
# VISUALISATIE
# =========================
output = image.copy()
overlay = image.copy()

for result in results:
    if result.masks is None:
        continue

    class_names = result.names

    for i, poly in enumerate(result.masks.xy):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        cls_id = int(result.boxes.cls[i].item()) if result.boxes is not None else -1
        score = float(result.boxes.conf[i].item()) if result.boxes is not None else 0.0

        # ✅ unieke kleur per segment (HSV → BGR voor betere spreiding)
        hue = int(180 * i / len(result.masks.xy))
        color = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
        )[0][0])

        # Polygon tekenen
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(output, [pts], True, color, 2)

        # Label
        if SHOW_LABELS and cls_id >= 0:
            x, y = pts[0][0]
            label = f"{class_names[cls_id]} {score:.2f}"
            cv2.putText(
                output,
                label,
                (int(x), max(20, int(y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

    # =========================
    # TIMING TONEN
    # =========================
    inference_time = result.speed['inference']  # ms

    cv2.putText(
        output,
        f"Inference: {inference_time:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        f"Total: {total_time:.1f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

# Transparantie toepassen
output = cv2.addWeighted(overlay, ALPHA, output, 1 - ALPHA, 0)

# =========================
# TONEN
# =========================
cv2.imshow("Segmentaties als polygonen", output)
cv2.waitKey(0)
cv2.destroyAllWindows()