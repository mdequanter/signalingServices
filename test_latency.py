"""
One-way latency test for the signaling server.

Measures how long a JSON message takes to travel:
    sender -> signaling server -> receiver

Default ("both" mode) runs the sender and the receiver in a single process,
so both endpoints share the same system clock and the one-way number is exact.

    python test_latency.py
    python test_latency.py --count 200 --interval 0.05
    python test_latency.py --server ws://192.168.0.81:9000 --no-tls
    python test_latency.py --payload 4096 --csv latency_oneway.csv

A 640x480 JPEG frame can be sent instead of a tiny JSON probe, to see what
latency the video path really gets:

    python test_latency.py --jpeg                     # base64 JPEG inside the JSON
    python test_latency.py --jpeg --binary            # raw JPEG bytes, like webcam_sender.py
    python test_latency.py --jpeg --quality 50 --resolution 320x240

Split mode measures across two machines. The one-way number is then only as
accurate as the clock sync between them (use NTP, otherwise a constant offset
is added to every sample):

    machine A:  python test_latency.py --role receiver
    machine B:  python test_latency.py --role sender
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import ssl
import statistics
import struct
import time

import websockets

DEFAULT_SERVER = "wss://signaling.ehb.be"
DEFAULT_ROOM = "/ws/pathnavigation"
SEND_TOKEN = "LTddk_ptxQX-omdw5B5rfpniA2wB-19KBxFaKuODMzw"
RECV_TOKEN = "B6zifTK3JWeH6E2tThPKLMwxt0QdqXVJ76GHfq7kTvs"

SESSION_ID = "latency-test"
TOPIC_NAME = "latency_probe"

# Same defaults as webcam_sender.py, so the numbers match the video path.
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
JPEG_QUALITY = 70
FALLBACK_IMAGE = "last_frame.jpg"

# Raw-binary probe header: double t_send + uint32 seq + uint8 warmup flag.
BIN_HEADER = ">dIB"
BIN_HEADER_SIZE = struct.calcsize(BIN_HEADER)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
}


def build_jpeg(width, height, quality, image_path=None):
    """Return the JPEG bytes of one `width`x`height` frame.

    Uses `image_path` when given, else last_frame.jpg, else a synthetic
    gradient-plus-noise frame that compresses like a real camera image.
    """

    try:
        import cv2
        import numpy as np
    except ImportError:
        raise SystemExit(
            "--jpeg needs opencv and numpy (pip install opencv-python numpy), "
            "or use --payload <bytes> to approximate the size instead."
        )

    source = image_path or (FALLBACK_IMAGE if os.path.exists(FALLBACK_IMAGE) else None)
    frame = cv2.imread(source) if source else None

    if frame is None:
        if image_path:
            raise SystemExit(f"Could not read image: {image_path}")
        # Diagonal gradient plus noise - detailed enough to compress realistically.
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        base = (xx / max(width - 1, 1) + yy / max(height - 1, 1)) * 127.5
        noise = np.random.default_rng(0).normal(0, 18, (height, width)).astype(np.float32)
        gray = np.clip(base + noise, 0, 255).astype(np.uint8)
        frame = cv2.merge([gray, np.roll(gray, 7, axis=1), np.roll(gray, 13, axis=0)])
    else:
        frame = cv2.resize(frame, (width, height))

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise SystemExit("JPEG encoding failed")

    return buf.tobytes()


async def connect(server, room, token, use_tls):

    uri = server.rstrip("/") + room
    headers = dict(HEADERS)
    headers["Authorization"] = f"Bearer {token}"

    return await websockets.connect(
        uri,
        ssl=ssl.create_default_context() if use_tls else None,
        origin="http://localhost",
        compression=None,
        additional_headers=headers,
        ping_interval=20,
        ping_timeout=20,
    )


# ---------------------------------------------------------------- sender side

async def run_sender(ws, count, interval, warmup, quiet,
                     padding="", jpeg=None, binary=False):
    """Send `warmup + count` probes, each stamped with its send time.

    `jpeg` bytes are sent raw with a timestamp header when `binary` is set
    (the shape webcam_sender.py uses), otherwise base64 inside the JSON probe.
    """

    total = warmup + count
    b64 = base64.b64encode(jpeg).decode("ascii") if (jpeg and not binary) else None

    for seq in range(total):
        is_warmup = seq < warmup

        if binary and jpeg:
            frame = struct.pack(BIN_HEADER, time.time(), seq, 1 if is_warmup else 0)
            await ws.send(frame + jpeg)
        else:
            data = {
                "name": TOPIC_NAME,
                "seq": seq,
                "warmup": is_warmup,
                "t_send": time.time(),
            }
            if b64 is not None:
                data["jpeg"] = b64
            elif padding:
                data["pad"] = padding
            await ws.send(json.dumps({
                "sessionId": SESSION_ID,
                "type": "topic",
                "from": "latency_sender",
                "to": "all",
                "data": data,
            }))

        if interval > 0 and seq < total - 1:
            await asyncio.sleep(interval)

    if not quiet:
        print(f"\n[sent] {total} messages ({warmup} warmup + {count} measured)")


# -------------------------------------------------------------- receiver side

async def run_receiver(ws, expected, timeout, quiet, samples):
    """Collect probes until `expected` measured samples arrive, or it goes quiet.

    `expected=None` means run until the connection idles out or Ctrl+C.
    """

    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            if not quiet:
                print(f"[idle] no message for {timeout:.1f}s - stopping")
            return
        except websockets.exceptions.ConnectionClosed:
            print("[warn] connection closed by server")
            return

        t_recv = time.time()

        if isinstance(raw, (bytes, bytearray)):
            # Raw binary probe: timestamp header followed by the JPEG payload.
            if len(raw) < BIN_HEADER_SIZE:
                continue
            wire_bytes = len(raw)
            t_send, seq, is_warmup = struct.unpack(BIN_HEADER, raw[:BIN_HEADER_SIZE])
        else:
            wire_bytes = len(raw.encode("utf-8"))
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            data = msg.get("data") or {}
            if msg.get("type") != "topic" or data.get("name") != TOPIC_NAME:
                continue  # room_joined, other traffic in the same room, ...

            t_send = data.get("t_send")
            if not isinstance(t_send, (int, float)):
                continue
            seq, is_warmup = data.get("seq"), data.get("warmup")

        latency_ms = (t_recv - t_send) * 1000.0

        if is_warmup:
            if not quiet:
                print(f"  warmup seq={seq}  {latency_ms:.2f} ms")
            continue

        samples.append({
            "seq": seq,
            "t_send": t_send,
            "t_recv": t_recv,
            "latency_ms": latency_ms,
            "bytes": wire_bytes,
        })

        if not quiet:
            print(f"  seq={seq:<4} {latency_ms:8.2f} ms  ({wire_bytes} bytes)")

        if expected is not None and len(samples) >= expected:
            return


# ------------------------------------------------------------------- reporting

def percentile(sorted_values, p):
    """Nearest-rank percentile on an already sorted list."""

    if not sorted_values:
        return float("nan")
    k = int(round((p / 100.0) * len(sorted_values) + 0.5)) - 1
    k = max(0, min(len(sorted_values) - 1, k))
    return sorted_values[k]


def report(samples, sent_count, csv_path):

    print("\n" + "=" * 52)
    print("ONE-WAY LATENCY  (sender -> server -> receiver)")
    print("=" * 52)

    if not samples:
        print("No samples received - is a peer connected to the same room?")
        return

    lat = sorted(s["latency_ms"] for s in samples)
    n = len(lat)

    print(f"samples      : {n}" + (f" / {sent_count} sent" if sent_count else ""))
    if sent_count:
        lost = sent_count - n
        print(f"lost         : {lost} ({lost / sent_count * 100:.1f}%)")
    print(f"message size : {samples[0]['bytes']} bytes")
    print(f"min          : {lat[0]:8.2f} ms")
    print(f"mean         : {statistics.fmean(lat):8.2f} ms")
    print(f"median (p50) : {statistics.median(lat):8.2f} ms")
    print(f"p95          : {percentile(lat, 95):8.2f} ms")
    print(f"p99          : {percentile(lat, 99):8.2f} ms")
    print(f"max          : {lat[-1]:8.2f} ms")

    if n > 1:
        print(f"stdev        : {statistics.stdev(lat):8.2f} ms")
        jitter = statistics.fmean(
            abs(samples[i]["latency_ms"] - samples[i - 1]["latency_ms"])
            for i in range(1, n)
        )
        print(f"jitter       : {jitter:8.2f} ms  (mean abs delta)")

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["seq", "t_send", "t_recv", "latency_ms", "bytes"]
            )
            writer.writeheader()
            writer.writerows(samples)
        print(f"\nWrote {len(samples)} samples to {csv_path}")


# ------------------------------------------------------------------ main modes

async def mode_both(args):

    uri = args.server.rstrip("/") + args.room
    print(f"Connecting receiver to {uri}")
    recv_ws = await connect(args.server, args.room, args.recv_token, args.tls)
    print(f"Connecting sender   to {uri}")
    send_ws = await connect(args.server, args.room, args.send_token, args.tls)
    print("Both peers connected - starting test\n")

    samples = []
    try:
        receiver = asyncio.create_task(
            run_receiver(recv_ws, args.count, args.timeout, args.quiet, samples)
        )
        await asyncio.sleep(0.3)  # let the receiver settle before the first probe
        await run_sender(send_ws, args.count, args.interval, args.warmup, args.quiet,
                         padding="x" * args.payload,
                         jpeg=args.jpeg_bytes, binary=args.binary)
        await receiver
    finally:
        await send_ws.close()
        await recv_ws.close()

    report(samples, args.count, args.csv)


async def mode_sender(args):

    print(f"Connecting sender to {args.server.rstrip('/') + args.room}")
    ws = await connect(args.server, args.room, args.send_token, args.tls)
    print("Connected\n")

    try:
        await run_sender(ws, args.count, args.interval, args.warmup, args.quiet,
                         padding="x" * args.payload,
                         jpeg=args.jpeg_bytes, binary=args.binary)
        await asyncio.sleep(0.5)  # let the last frames flush
    finally:
        await ws.close()

    print("Read the results on the receiver side.")


async def mode_receiver(args):

    print(f"Connecting receiver to {args.server.rstrip('/') + args.room}")
    ws = await connect(args.server, args.room, args.recv_token, args.tls)
    print("Connected - waiting for probes (Ctrl+C to stop)")
    print("Split mode: accuracy depends on clock sync between both machines.\n")

    samples = []
    try:
        await run_receiver(ws, None, args.timeout, args.quiet, samples)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await ws.close()

    report(samples, None, args.csv)


def main():

    parser = argparse.ArgumentParser(
        description="Measure one-way message latency over the signaling server."
    )
    parser.add_argument("--role", choices=["both", "sender", "receiver"], default="both",
                        help="both = sender and receiver in one process, one clock (default)")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"signaling server (default: {DEFAULT_SERVER})")
    parser.add_argument("--room", default=DEFAULT_ROOM,
                        help=f"room path (default: {DEFAULT_ROOM})")
    parser.add_argument("--send-token", default=SEND_TOKEN, help="bearer token for the sender")
    parser.add_argument("--recv-token", default=RECV_TOKEN, help="bearer token for the receiver")
    parser.add_argument("--no-tls", dest="tls", action="store_false",
                        help="plain ws:// instead of wss://")
    parser.add_argument("--count", type=int, default=50,
                        help="measured messages (default: 50)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="extra messages sent first and excluded (default: 3)")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="seconds between messages (default: 0.1)")
    parser.add_argument("--payload", type=int, default=0,
                        help="extra padding bytes per message (default: 0)")
    parser.add_argument("--jpeg", action="store_true",
                        help=f"send a real {FRAME_WIDTH}x{FRAME_HEIGHT} JPEG frame as the payload")
    parser.add_argument("--image", default=None,
                        help=f"source image for --jpeg (default: {FALLBACK_IMAGE}, else synthetic)")
    parser.add_argument("--resolution", default=f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                        help=f"frame size for --jpeg (default: {FRAME_WIDTH}x{FRAME_HEIGHT})")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY,
                        help=f"JPEG quality for --jpeg (default: {JPEG_QUALITY})")
    parser.add_argument("--binary", action="store_true",
                        help="send the JPEG as raw bytes like webcam_sender.py, not base64 JSON")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="give up after this many idle seconds (default: 5)")
    parser.add_argument("--csv", default=None,
                        help="write per-message samples to this CSV")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")

    args = parser.parse_args()

    args.jpeg_bytes = None
    if args.jpeg or args.image:
        try:
            w, h = (int(v) for v in args.resolution.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--resolution must look like 640x480, got: {args.resolution}")

        args.jpeg_bytes = build_jpeg(w, h, args.quality, args.image)
        wire = len(args.jpeg_bytes) if args.binary else (len(args.jpeg_bytes) + 2) // 3 * 4
        print(f"Frame: {w}x{h} q{args.quality} -> {len(args.jpeg_bytes)} bytes JPEG, "
              f"{wire} bytes on the wire "
              f"({'raw binary' if args.binary else 'base64 in JSON'})")
    elif args.binary:
        raise SystemExit("--binary only applies together with --jpeg")

    runner = {"both": mode_both, "sender": mode_sender, "receiver": mode_receiver}[args.role]
    try:
        asyncio.run(runner(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
