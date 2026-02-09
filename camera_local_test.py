"""Minimal local camera test viewer.

Usage:
  python camera_local_test.py [device] [width] [height]

Defaults: device=0, width=640, height=480

This script captures frames from the local camera and displays them in a window.
It prints the current FPS every second to help verify performance.
"""

import sys
import time

import cv2

DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 640
HEIGHT = int(sys.argv[3]) if len(sys.argv) > 3 else 480

def main():
    cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera device {DEVICE}")
        return

    # try to set resolution (may be ignored by some cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    win = "Camera Test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: failed to read frame")
                break

            cv2.imshow(win, frame)
            frame_count += 1

            # print FPS each second
            now = time.time()
            if now - t0 >= 1.0:
                print(f"FPS: {frame_count}")
                frame_count = 0
                t0 = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()