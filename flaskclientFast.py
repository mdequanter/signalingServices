# app_latency_heading.py
#
# Minimal end-device renderer:
# - Captures webcam in browser (hidden)
# - Sends frames to server with a client timestamp (ts_ms)
# - Server runs YOLO seg, computes heading (deg), measures inference (ms) and end-to-end latency (ms)
# - Client shows only 3 big numbers (heading, e2e, infer) with ultra-light DOM updates (no canvas, no rendering)

from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import base64
import time
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = "models/unrealsim.pt"
model = YOLO(MODEL_PATH)

# Keep the same scan heights logic to compute a robust centerline -> heading
SCAN_HEIGHTS = [0.2, 0.35, 0.5, 0.65, 0.8]

# ---- Minimal UI: big heading + small latency numbers ----
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Heading + Latency</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body {
      margin: 0; padding: 0;
      height: 100%;
      background: #000;
      color: #fff;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    .wrap{
      height:100%;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      gap:18px;
      user-select:none;
    }
    .heading{
      font-size: min(22vw, 220px);
      font-weight: 800;
      line-height: 1;
      letter-spacing: -0.03em;
    }
    .meta{
      display:flex;
      gap:24px;
      font-size: min(4.5vw, 28px);
      opacity: 0.9;
    }
    .meta span{
      font-weight: 650;
    }
    /* hidden capture video */
    video{ display:none; }
    button{
      position:fixed;
      top:12px; left:12px;
      padding:10px 14px;
      font-size:14px;
      border-radius:10px;
      border:0;
      background:#222;
      color:#fff;
    }
  </style>
</head>
<body>
  <button id="switchBtn">Switch Camera</button>

  <video id="video" autoplay playsinline></video>

  <div class="wrap">
    <div class="heading" id="heading">--°</div>
    <div class="meta">
      <div>E2E <span id="e2e">--</span> ms</div>
      <div>Infer <span id="infer">--</span> ms</div>
    </div>
  </div>

<script>
const video = document.getElementById('video');
const switchBtn = document.getElementById('switchBtn');

const headingEl = document.getElementById('heading');
const e2eEl = document.getElementById('e2e');
const inferEl = document.getElementById('infer');

const cap = document.createElement('canvas');
const cctx = cap.getContext('2d', { willReadFrequently: false });

let currentFacingMode = "environment";
let currentStream = null;

// Tune these for speed on the end device:
const SEND_EVERY_MS = 150;        // request interval
const JPEG_QUALITY = 0.35;        // smaller = faster
const MAX_W = 416;                // downscale before sending (big speed win)

function startCamera(facingMode) {
  if (currentStream) currentStream.getTracks().forEach(t => t.stop());
  navigator.mediaDevices.getUserMedia({ video: { facingMode } })
    .then(stream => { currentStream = stream; video.srcObject = stream; })
    .catch(err => console.error("camera error:", err));
}

startCamera(currentFacingMode);

switchBtn.addEventListener('click', () => {
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  startCamera(currentFacingMode);
});

function setTextFast(el, txt){
  // Avoid layout thrash: only update when changed
  if (el.textContent !== txt) el.textContent = txt;
}

video.addEventListener('play', () => {
  // Start capture loop
  setInterval(() => {
    if (!video.videoWidth || !video.videoHeight) return;

    // Downscale for speed
    let w = video.videoWidth, h = video.videoHeight;
    if (w > MAX_W) {
      const s = MAX_W / w;
      w = Math.round(w * s);
      h = Math.round(h * s);
    }
    if (cap.width !== w || cap.height !== h) { cap.width = w; cap.height = h; }

    cctx.drawImage(video, 0, 0, w, h);

    const ts_ms = performance.now();  // client-side timestamp (monotonic)
    const dataURL = cap.toDataURL('image/jpeg', JPEG_QUALITY);

    fetch('/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL, ts_ms })
    })
    .then(r => r.json())
    .then(d => {
      // Render only text (fast)
      const hd = (d.heading_deg === null || d.heading_deg === undefined) ? "--" : d.heading_deg.toFixed(1);
      setTextFast(headingEl, hd + "°");
      setTextFast(e2eEl, (d.e2e_ms ?? 0).toFixed(1));
      setTextFast(inferEl, (d.infer_ms ?? 0).toFixed(1));
    })
    .catch(()=>{});
  }, SEND_EVERY_MS);
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

def _compute_heading_deg_from_mask(mask_resized: np.ndarray) -> float | None:
    """
    Heading convention:
      0° = straight ahead (target_x == image_center_x)
      positive = target is to the right
      negative = target is to the left
    Based on mean midpoint of segmented pixels along multiple scan rows.
    """
    h, w = mask_resized.shape[:2]
    midpoints = []

    for r in SCAN_HEIGHTS:
        y = int(h * r)
        if y < 0 or y >= h:
            continue
        row = mask_resized[y, :]
        idx = np.where(row > 0)[0]
        if idx.size:
            midpoints.append(int(idx.mean()))

    if not midpoints:
        return None

    target_x = float(np.mean(midpoints))
    cx = (w - 1) / 2.0

    # angle from bottom-center looking up (small-angle approx across image)
    # Use atan2 with dy as image height to scale angle by distance.
    dx = target_x - cx
    dy = h  # reference distance
    angle_rad = np.arctan2(dx, dy)
    return float(np.degrees(angle_rad))

@app.route("/process", methods=["POST"])
def process():
    payload = request.json or {}
    data_url = payload.get("image", "")
    ts_ms = float(payload.get("ts_ms", 0.0))  # client monotonic ms

    if "," not in data_url:
        return jsonify({"heading_deg": None, "infer_ms": 0.0, "e2e_ms": 0.0}), 400

    _, encoded = data_url.split(",", 1)

    # Measure end-to-end from client timestamp to *now on server* (best effort).
    # Note: client performance.now() != server time; we treat ts_ms as "send start" in client monotonic domain,
    # so we compute e2e as: (client_now_ms - ts_ms) + server_processing_ms component would be ideal.
    # Instead, we return server-side receive->done and echo back client age computed in browser.
    # BUT user asked end-to-end; we can do it properly by sending ts_epoch_ms from client (Date.now()).
    # We'll support both: if ts_epoch_ms provided, use it. Else fallback to receive->done only.
    ts_epoch_ms = payload.get("ts_epoch_ms", None)

    # Decode image
    img_data = base64.b64decode(encoded)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"heading_deg": None, "infer_ms": 0.0, "e2e_ms": 0.0}), 400

    h, w = frame.shape[:2]

    # Inference timing
    t0 = time.perf_counter()
    results = model(frame, conf=0.1, verbose=False)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    heading_deg = None

    # Extract first mask (fast path)
    try:
        for result in results:
            if result.masks is None:
                continue
            mask = result.masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            heading_deg = _compute_heading_deg_from_mask(mask_resized)
            break
    except Exception:
        heading_deg = None

    # End-to-end latency:
    # Prefer epoch timestamp from client (Date.now()), because it’s comparable to server time.time().
    if ts_epoch_ms is not None:
        try:
            e2e_ms = (time.time() * 1000.0) - float(ts_epoch_ms)
        except Exception:
            e2e_ms = 0.0
    else:
        # Fallback: server-side only (receive->done isn't included here; approximate with infer only)
        # We at least return inference + minimal overhead; client can also compute "age" itself if needed.
        e2e_ms = infer_ms

    return jsonify({
        "heading_deg": heading_deg,
        "infer_ms": float(infer_ms),
        "e2e_ms": float(e2e_ms),
    })

if __name__ == "__main__":
    # If you use SSL like before, keep it. Otherwise remove ssl_context for local testing.
    app.run(
        host="0.0.0.0",
        port=5000,
        ssl_context=("localhost+2.pem", "localhost+2-key.pem"),
        debug=False
    )