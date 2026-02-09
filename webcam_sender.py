import asyncio
import time
import cv2
import websockets

SIGNALING_SERVER = "ws://192.168.0.74:9000"

TARGET_WIDTH, TARGET_HEIGHT = 640, 480
FPS = 8
JPEG_QUALITY = 70

def open_camera_best_effort(max_index=8):
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        # Force MJPG (vaak het verschil tussen timeout en OK)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Probeer snel frames te lezen
        ok = False
        t0 = time.time()
        while time.time() - t0 < 1.0:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                break
            time.sleep(0.05)

        if ok:
            return idx, cap

        cap.release()

    return None, None

async def webcam_sender():
    cam_idx, cap = open_camera_best_effort(max_index=8)
    if cap is None:
        raise RuntimeError("Geen werkende camera gevonden op /dev/video0..7")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
    frame_interval = 1.0 / float(FPS)
    next_t = time.time()

    async with websockets.connect(SIGNALING_SERVER, max_size=None) as ws:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                await asyncio.sleep(0.05)
                continue

            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                await ws.send(jpg.tobytes())

            # throttle
            now = time.time()
            sleep_for = next_t - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            next_t += frame_interval

    cap.release()

asyncio.run(webcam_sender())
