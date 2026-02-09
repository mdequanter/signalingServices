"""Viewer client: receive camera frames via WebSocket and display them with OpenCV.

Usage:
  python view_camera.py [host] [port]

Defaults: host=127.0.0.1, port=9000

Controls:
- Press 'q' in the window to quit.

Requires:
  pip install websockets opencv-python numpy
"""

import sys
import asyncio
import json
import base64
from typing import Optional

import cv2
import numpy as np
import websockets

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
URI = f"ws://{HOST}:{PORT}"
WINDOW = "Camera"

async def main():
    print(f"Connecting to {URI}")
    async with websockets.connect(URI, max_size=None) as ws:
        print("Connected")
        count = 0
        try:
            async for msg in ws:
                try:
                    obj = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "broadcast":
                    continue
                data = obj.get("data") or {}
                if data.get("type") != "frame":
                    continue
                b64 = data.get("jpeg")
                if not b64:
                    continue
                try:
                    raw = base64.b64decode(b64)
                    arr = np.frombuffer(raw, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    cv2.imshow(WINDOW, frame)
                    count += 1
                    if count % 30 == 0:
                        print(f"Received frames: {count}")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    continue
        except KeyboardInterrupt:
            pass
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
