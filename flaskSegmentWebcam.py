# app.py
from flask import Flask, render_template_string, request, jsonify, Response
import cv2
import numpy as np
import base64
import time
import threading
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = "models/unrealsim.pt"
model = YOLO(MODEL_PATH)

SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# ---- Debug stream (MJPEG) shared state ----
DEBUG_STREAM_ENABLED = True
debug_lock = threading.Lock()
latest_debug_jpeg: bytes | None = None
latest_debug_ts: float = 0.0  # optional timestamp

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Arrow Only</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: transparent;
    }
    /* Fullscreen overlay canvas */
    #overlay {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      background: transparent;
      display: block;
    }
    /* Video is only used for capture (hidden) */
    #video { display: none; }

    #switchBtn {
      position: fixed;
      top: 12px;
      left: 12px;
      z-index: 10;
      padding: 10px 14px;
      font-size: 14px;
    }

    /* Optional: show debug stream small preview */
    #dbg {
      position: fixed;
      right: 12px;
      top: 12px;
      width: 360px;
      border: 2px solid rgba(0,0,0,0.5);
      border-radius: 8px;
      z-index: 10;
      background: #000;
    }
    #dbgLabel {
      position: fixed;
      right: 12px;
      top: 12px;
      transform: translateY(-110%);
      color: white;
      font-family: sans-serif;
      font-size: 12px;
      background: rgba(0,0,0,0.6);
      padding: 4px 8px;
      border-radius: 6px;
      z-index: 10;
    }
  </style>
</head>
<body>
  <button id="switchBtn">Switch Camera</button>

  <!-- Hidden capture video -->
  <video id="video" autoplay playsinline></video>

  <!-- Fullscreen arrow overlay -->
  <canvas id="overlay"></canvas>

  <!-- Optional: debug preview -->
  <div id="dbgLabel">debug_stream</div>
  <img id="dbg" src="/debug_stream" />

<script>
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const switchBtn = document.getElementById('switchBtn');

// Offscreen canvas for capture
const cap = document.createElement('canvas');
const cctx = cap.getContext('2d');

let currentFacingMode = "user";
let currentStream = null;

function resizeOverlay() {
  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
}
window.addEventListener('resize', resizeOverlay);
resizeOverlay();

function startCamera(facingMode) {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
  }
  navigator.mediaDevices.getUserMedia({
    video: { facingMode }
  }).then(stream => {
    currentStream = stream;
    video.srcObject = stream;
  }).catch(err => console.error("Error accessing camera:", err));
}
startCamera(currentFacingMode);

switchBtn.addEventListener('click', () => {
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  startCamera(currentFacingMode);
});

video.addEventListener('play', () => {
  cap.width = video.videoWidth;
  cap.height = video.videoHeight;

  setInterval(() => {
    // capture frame offscreen
    cctx.drawImage(video, 0, 0, cap.width, cap.height);
    const dataURL = cap.toDataURL('image/jpeg', 0.5);

    fetch('/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    })
    .then(r => r.json())
    .then(data => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + data.image;
      img.onload = () => {
        // draw ONLY arrow png (transparent background)
        octx.clearRect(0, 0, overlay.width, overlay.height);
        octx.drawImage(img, 0, 0, overlay.width, overlay.height);
      };
    })
    .catch(err => console.error(err));
  }, 300);
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/process_frame", methods=["POST"])
def process_frame():
    content = request.json
    data_url = content["image"]
    _, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)

    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"image": ""}), 400

    height, width = frame.shape[:2]

    # 1) Timed inference
    t0 = time.perf_counter()
    results = model(frame, conf=0.1, verbose=False)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000.0

    # 2) Client output: transparent BGRA with ONLY arrow
    arrow_only = np.zeros((height, width, 4), dtype=np.uint8)  # BGRA

    # 3) Server debug view: frame + seg + scanlines + arrow + timing
    debug_vis = frame.copy()
    midpoints = []

    for result in results:
        if result.masks is None:
            continue

        # Use first mask (same as your original code)
        mask = result.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # green segmentation overlay on debug image
        green_overlay = np.full_like(debug_vis, (0, 255, 0))
        blended = cv2.addWeighted(debug_vis, 0.3, green_overlay, 0.7, 0)
        debug_vis[mask_resized > 0] = blended[mask_resized > 0]

        # scanlines + midpoints
        for r in SCAN_HEIGHTS:
            y = int(height * r)
            if y >= height:
                continue

            scan_row = mask_resized[y, :]
            indices = np.where(scan_row > 0)[0]
            if len(indices) > 0:
                midpoint_x = int(np.mean(indices))
                midpoints.append((midpoint_x, y))
                cv2.circle(debug_vis, (midpoint_x, y), 5, (255, 0, 0), -1)

            cv2.line(debug_vis, (0, y), (width, y), (150, 150, 150), 1)

    # arrow on both outputs
    if midpoints:
        avg_x = int(np.mean([pt[0] for pt in midpoints]))
        target_point = (avg_x, min([pt[1] for pt in midpoints]))
        start_point = (width // 2, height)

        # debug arrow (BGR)
        cv2.arrowedLine(debug_vis, start_point, target_point, (0, 0, 255), 5, tipLength=0.2)
        # client arrow (BGRA)
        cv2.arrowedLine(arrow_only, start_point, target_point, (0, 0, 255, 255), 8, tipLength=0.25)

    # inference time text on debug image (outlined)
    text = f"Inference: {infer_ms:.1f} ms"
    cv2.putText(debug_vis, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(debug_vis, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # 4) Update MJPEG buffer for /debug_stream (headless-friendly)
    if DEBUG_STREAM_ENABLED:
        # Optional: resize for performance (uncomment if needed)
        # debug_vis = cv2.resize(debug_vis, (width // 2, height // 2))

        ok, jpg = cv2.imencode(".jpg", debug_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            global latest_debug_jpeg, latest_debug_ts
            with debug_lock:
                latest_debug_jpeg = jpg.tobytes()
                latest_debug_ts = time.time()

    # 5) Return to client: transparent PNG with ONLY arrow
    ok, png = cv2.imencode(".png", arrow_only)
    if not ok:
        return jsonify({"image": ""}), 500
    encoded_result = base64.b64encode(png).decode("utf-8")
    return jsonify({"image": encoded_result})


@app.route("/debug_stream")
def debug_stream():
    def gen():
        boundary = b"--frame"
        while True:
            with debug_lock:
                frame = latest_debug_jpeg

            if frame is None:
                time.sleep(0.05)
                continue

            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                + frame + b"\r\n"
            )

            # stream FPS (adjust as desired)
            time.sleep(0.05)  # ~20 fps

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # Tip: bij Flask debug reloader kunnen OpenCV/windows/threads raar doen.
    # Als je issues ziet: zet debug=False of use_reloader=False.
    app.run(
        host="0.0.0.0",
        port=5000,
        ssl_context=("localhost+2.pem", "localhost+2-key.pem"),
        debug=True,
        use_reloader=False
    )