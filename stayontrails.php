<?php
// index.php
?><!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Webcam → WebSocket Sender</title>
  <style>
    html,body{margin:0;height:100%;background:#000;color:#fff;font-family:system-ui}
    .wrap{position:relative;z-index:2;min-height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:16px;box-sizing:border-box}
    .big{font-size:28px;font-weight:700}
    .row{opacity:.9}
    .compassWrap{display:flex;flex-direction:column;align-items:center;gap:6px}
    #compass{width:140px;height:140px;display:block}
    .panel{background:rgba(0,0,0,.45);backdrop-filter:blur(3px);padding:14px 16px;border-radius:12px;border:1px solid rgba(255,255,255,.15);display:flex;flex-direction:column;align-items:center;gap:10px}
    .controls{display:flex;gap:10px}
    button{padding:10px 14px;border-radius:10px;border:0;background:#222;color:#fff;font-size:14px}
    #video{position:fixed;inset:0;width:100vw;height:100vh;object-fit:cover;z-index:0;background:#000}
    #cap{display:none;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <div class="big" id="status">Idle</div>
    <div class="row">Sent: <span id="sent">0</span> frames <span id="kbps">0</span> kbps</div>
    <div class="row">Errors: <span id="errs">0</span></div>
    <div class="row">Latency: <span id="latency">--</span> ms</div>
    <div class="compassWrap">
      <canvas id="compass" width="140" height="140"></canvas>
      <div class="row">Heading: <span id="heading">--</span>&deg;</div>
    </div>
    <div class="controls">
      <button id="btn">Start</button>
      <button id="switchCam" disabled>Switch Camera</button>
    </div>
  </div>
</div>

<video id="video" autoplay playsinline></video>
<canvas id="cap"></canvas>

<script>
(() => {
  const SIGNALING_SERVER = "wss://signaling.ehb.be";

  const TARGET_W = 640, TARGET_H = 480;
  const FPS = 8;
  const JPEG_QUALITY = 0.70; // 0..1 (browser)
  const INTERVAL_MS = Math.round(1000 / FPS);

  const statusEl = document.getElementById("status");
  const sentEl = document.getElementById("sent");
  const kbpsEl = document.getElementById("kbps");
  const errsEl = document.getElementById("errs");
  const latencyEl = document.getElementById("latency");
  const headingEl = document.getElementById("heading");
  const btn = document.getElementById("btn");
  const switchCamBtn = document.getElementById("switchCam");

  const video = document.getElementById("video");
  const cap = document.getElementById("cap");
  const ctx = cap.getContext("2d", { alpha: false });
  const compass = document.getElementById("compass");
  const compCtx = compass.getContext("2d");

  let ws = null;
  let stream = null;
  let activeVideoDeviceId = null;
  let availableVideoInputs = [];
  let timer = null;
  let sentFrames = 0;
  let errors = 0;
  let latestHeading = null;
  let nextFrameId = 1;
  const sentAtByFrameId = new Map();

  // bitrate calc
  let bytesSince = 0;
  let lastRateT = performance.now();

  // =========================
  // AUDIO (MP3) + HEADING → CMD
  // =========================
  // Plaats je files op: /audio/left.mp3, /audio/right.mp3, /audio/forward.mp3
  const SOUND_MAP = {
    left:    "audio/left.mp3",
    right:   "audio/right.mp3",
    forward: "audio/forward.mp3",
    started: "audio/application_started.mp3",
  };

  const player = new Audio();
  player.preload = "auto";

  // optioneel preload (snellere respons)
  const preloaded = {};
  for (const [k, url] of Object.entries(SOUND_MAP)) {
    const a = new Audio(url);
    a.preload = "auto";
    preloaded[k] = a;
  }

  let audioEnabled = false;   // wordt true na Start-klik en WS open
  let lastCmd = null;
  let lastCmdAt = 0;

  let targetHeading = null;   // "forward" referentie; wordt gezet bij eerste heading

  const FORWARD_DEG = 12;     // binnen ±12° = forward
  const COOLDOWN_MS = 5000;    // minimaal 0.9s tussen uitspreken

  function angleDiffDeg(current, target) {
    // resultaat in [-180, +180]
    return (current - target + 540) % 360 - 180;
  }

  function headingToCmd(heading) {
    if (targetHeading === null) return null;
    const d = angleDiffDeg(heading, targetHeading);
    if (Math.abs(d) <= FORWARD_DEG) return "forward";
    return d > 0 ? "right" : "left";
  }

  async function playCmd(cmd) {
    if (!audioEnabled) return;
    if (!cmd || !SOUND_MAP[cmd]) return;

    const now = performance.now();
    if (cmd === lastCmd && (now - lastCmdAt) < COOLDOWN_MS) return;

    lastCmd = cmd;
    lastCmdAt = now;

    try {
      player.pause();
      player.currentTime = 0;
      player.src = SOUND_MAP[cmd];
      await player.play();
    } catch (e) {
      // iOS/Safari of browser policies kunnen play blokkeren zonder user gesture
      console.log("Audio play blocked/failed:", e);
    }
  }
  // =========================
  // /AUDIO
  // =========================

  function drawArrow(headingDeg) {
    const w = compass.width;
    const h = compass.height;
    const cx = w / 2;
    const cy = h / 2;

    compCtx.clearRect(0, 0, w, h);

    compCtx.beginPath();
    compCtx.arc(cx, cy, 62, 0, Math.PI * 2);
    compCtx.strokeStyle = "rgba(255,255,255,0.35)";
    compCtx.lineWidth = 2;
    compCtx.stroke();

    compCtx.fillStyle = "rgba(255,255,255,0.7)";
    compCtx.font = "12px system-ui";
    compCtx.textAlign = "center";
    compCtx.fillText("N", cx, 14);

    if (typeof headingDeg !== "number" || Number.isNaN(headingDeg)) return;

    const angleRad = (-headingDeg * Math.PI) / 180;
    compCtx.save();
    compCtx.translate(cx, cy);
    compCtx.rotate(angleRad);

    compCtx.beginPath();
    compCtx.moveTo(-8, -3);
    compCtx.lineTo(40, -3);
    compCtx.lineTo(40, -10);
    compCtx.lineTo(56, 0);
    compCtx.lineTo(40, 10);
    compCtx.lineTo(40, 3);
    compCtx.lineTo(-8, 3);
    compCtx.closePath();
    compCtx.fillStyle = "#ff3b30";
    compCtx.fill();
    compCtx.restore();
  }

  function normalizeHeading(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return ((n % 360) + 360) % 360;
  }

  function setStatus(s){ statusEl.textContent = s; }
  function incErr(){
    errors++;
    errsEl.textContent = String(errors);
  }

  async function refreshVideoInputs() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    availableVideoInputs = devices.filter(d => d.kind === "videoinput");
    switchCamBtn.disabled = availableVideoInputs.length < 2 || !stream;
  }

  async function startCamera(preferredDeviceId = null) {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    const videoConstraints = preferredDeviceId
      ? {
          deviceId: { exact: preferredDeviceId },
          width: { ideal: TARGET_W },
          height: { ideal: TARGET_H }
        }
      : {
          width: { ideal: TARGET_W },
          height: { ideal: TARGET_H },
          facingMode: "environment"
        };

    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraints,
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    const track = stream.getVideoTracks()[0];
    activeVideoDeviceId = track?.getSettings?.().deviceId ?? null;
    await refreshVideoInputs();
  }

  async function start() {
    btn.disabled = true;
    setStatus("Requesting camera");

    try {
      await startCamera(activeVideoDeviceId);
    } catch (e) {
      incErr();
      setStatus("Camera error");
      console.error(e);
      btn.disabled = false;
      return;
    }

    // fixed capture size (fast + predictable)
    cap.width = TARGET_W;
    cap.height = TARGET_H;

    setStatus("Connecting WS");

    try {
      ws = new WebSocket(SIGNALING_SERVER);
      ws.binaryType = "arraybuffer";

      ws.onopen = async () => {
        setStatus("Streaming");
        btn.disabled = false;

        // AUDIO: enable + reset heading reference
        audioEnabled = true;
        targetHeading = null;
        lastCmd = null;
        lastCmdAt = 0;
        playCmd("started");

        // Wait 5 seconds after start before sending frames
        await new Promise(resolve => setTimeout(resolve, 5000));
        if (!ws || ws.readyState !== WebSocket.OPEN || timer) return;

        // Start send loop
        timer = setInterval(captureAndSend, INTERVAL_MS);
      };

      ws.onerror = (e) => {
        incErr();
        console.error("WS error", e);
      };

      ws.onclose = () => {
        setStatus("WS closed");
        stop(false);
      };

      ws.onmessage = (msg) => {
        let heading = null;
        let frameId = null;

        if (typeof msg.data === "string") {
          try {
            const payload = JSON.parse(msg.data);
            heading = payload?.heading;
            frameId = payload?.frame_id;
          } catch {
            heading = msg.data;
          }
        }

        const normalized = normalizeHeading(heading);
        if (normalized !== null) {
          latestHeading = normalized;
          headingEl.textContent = normalized.toFixed(1);
          drawArrow(latestHeading);

          // ===== HEADING → AUDIO =====
          if (targetHeading === null) {
            // Eerste geldige heading wordt "forward" referentie
            targetHeading = latestHeading;
          } else {
            const cmd = headingToCmd(latestHeading);
            playCmd(cmd);
          }
          // ===========================
        }

        if (frameId !== null && frameId !== undefined) {
          const id = String(frameId);
          const sentAt = sentAtByFrameId.get(id);
          if (typeof sentAt === "number") {
            const latencyMs = performance.now() - sentAt;
            latencyEl.textContent = latencyMs.toFixed(1);
            sentAtByFrameId.delete(id);
          }
        }
      };

    } catch (e) {
      incErr();
      setStatus("WS connect failed");
      console.error(e);
      stop(true);
      return;
    }
  }

  function stop(allowButton=true) {
    if (timer) { clearInterval(timer); timer = null; }
    if (ws) {
      try { ws.close(); } catch {}
      ws = null;
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    switchCamBtn.disabled = true;
    sentAtByFrameId.clear();
    nextFrameId = 1;

    // AUDIO reset
    audioEnabled = false;
    targetHeading = null;
    lastCmd = null;
    try { player.pause(); player.currentTime = 0; } catch {}

    if (allowButton) {
      btn.disabled = false;
      btn.textContent = "Start";
    }
  }

  function updateRate(bytesJustSent) {
    bytesSince += bytesJustSent;
    const now = performance.now();
    const dt = now - lastRateT;
    if (dt >= 1000) {
      const kbitsPerSec = (bytesSince * 8) / dt;
      kbpsEl.textContent = kbitsPerSec.toFixed(1);
      bytesSince = 0;
      lastRateT = now;
    }
  }

  function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !video.videoHeight) return;

    // Draw current frame to canvas (scaled)
    ctx.drawImage(video, 0, 0, TARGET_W, TARGET_H);

    // Encode JPEG -> Blob -> ArrayBuffer (binary)
    cap.toBlob(async (blob) => {
      if (!blob) return;
      try {
        const frameId = String(nextFrameId++);
        const buf = await blob.arrayBuffer();
        ws.send(JSON.stringify({ type: "frame_meta", frame_id: frameId }));
        sentAtByFrameId.set(frameId, performance.now());
        ws.send(buf);

        sentFrames++;
        sentEl.textContent = String(sentFrames);
        updateRate(buf.byteLength);

        if (sentAtByFrameId.size > 200) {
          const cutoff = performance.now() - 5000;
          for (const [id, t] of sentAtByFrameId) {
            if (t < cutoff) sentAtByFrameId.delete(id);
          }
        }
      } catch (e) {
        incErr();
        console.error(e);
      }
    }, "image/jpeg", JPEG_QUALITY);
  }

  btn.addEventListener("click", () => {
    if (timer || (ws && ws.readyState === WebSocket.OPEN)) {
      setStatus("Stopping");
      stop(true);
    } else {
      btn.textContent = "Stop";
      start();
    }
  });

  switchCamBtn.addEventListener("click", async () => {
    if (!stream) {
      setStatus("Start eerst");
      return;
    }

    try {
      await refreshVideoInputs();
      if (availableVideoInputs.length < 2) {
        setStatus("Geen extra camera gevonden");
        return;
      }

      let idx = availableVideoInputs.findIndex(d => d.deviceId === activeVideoDeviceId);
      if (idx < 0) idx = 0;
      const next = availableVideoInputs[(idx + 1) % availableVideoInputs.length];
      await startCamera(next.deviceId);

      if (ws && ws.readyState === WebSocket.OPEN) {
        setStatus("Streaming");
      }
    } catch (e) {
      incErr();
      console.error("Camera switch error", e);
      setStatus("Camera switch error");
    }
  });

  drawArrow(null);

})();
</script>
</body>
</html>
