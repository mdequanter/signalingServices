import asyncio
import time
import cv2
import websockets

SIGNALING_SERVER = "ws://192.168.0.74:9000"

CAM_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 640, 480
JPEG_QUALITY = 70          # 1..100
FPS = 8                    # target send rate

async def webcam_sender():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Kon webcam niet openen")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
    frame_interval = 1.0 / float(FPS)

    async with websockets.connect(SIGNALING_SERVER, max_size=None) as ws:
        next_t = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Zorg dat resolutie consistent is (optioneel)
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            # Stuur raw JPEG bytes (receiver decode_message_to_frame kan dit meteen)
            await ws.send(jpg.tobytes())

            # (Optioneel) lokaal tonen + stoppen met 'q'
            cv2.imshow("Webcam Sender (local preview)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Throttle naar FPS
            now = time.time()
            sleep_for = next_t - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            next_t += frame_interval

    cap.release()
    cv2.destroyAllWindows()

asyncio.run(webcam_sender())
