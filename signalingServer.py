import asyncio
import csv
import os
import websockets
from collections import defaultdict

TOKENS_CSV = "authtokens.csv"
_valid_tokens = set()
_tokens_mtime = None
_tokens_lock = asyncio.Lock()

# room_id -> set(websocket)
rooms = defaultdict(set)
rooms_lock = asyncio.Lock()

def _read_tokens(csv_path: str) -> set[str]:
    tokens = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            token = (row.get("token") or "").strip().strip('"')
            if token:
                tokens.add(token)
    return tokens

async def get_valid_tokens() -> set[str]:
    global _valid_tokens, _tokens_mtime
    mtime = os.path.getmtime(TOKENS_CSV)

    async with _tokens_lock:
        if _tokens_mtime != mtime:
            _valid_tokens = _read_tokens(TOKENS_CSV)
            _tokens_mtime = mtime
            print(f"🔄 Reloaded {len(_valid_tokens)} tokens from {TOKENS_CSV}")
        return _valid_tokens

def room_from_path(path: str) -> str:
    """
    Verwacht paden zoals:
      /ws/demo-session-001
      /room/demo
    """
    path = (path or "/").split("?", 1)[0]
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] in ("ws", "room"):
        return parts[1]
    # fallback: alles in default room
    return "default"

async def process_request(path, request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        return (401, [], b"Missing/invalid Authorization header")

    token = auth.split(" ", 1)[1].strip()

    valid = await get_valid_tokens()
    if token not in valid:
        return (403, [], b"Invalid token")

    # (optioneel) je kan hier ook room naming rules afdwingen
    return None

async def signaling(websocket):
    # websockets >= 12: path zit in websocket.request.path
    path = websocket.request.path if hasattr(websocket, "request") else "/"
    room_id = room_from_path(path)

    async with rooms_lock:
        rooms[room_id].add(websocket)
        peer_count = len(rooms[room_id])

    # (optioneel) informeer de client
    try:
        await websocket.send(f'{{"type":"room_joined","room":"{room_id}","peers":{peer_count}}}')
    except Exception:
        pass

    print(f"✅ Client joined room '{room_id}' from {websocket.remote_address} (peers={peer_count})")

    try:
        async for message in websocket:
            # Forward ONLY to peers in same room
            async with rooms_lock:
                targets = [ws for ws in rooms[room_id] if ws != websocket]

            # kleine optimalisatie: gather
            if targets:
                await asyncio.gather(*(t.send(message) for t in targets), return_exceptions=True)

    finally:
        async with rooms_lock:
            rooms[room_id].discard(websocket)
            remaining = len(rooms[room_id])
            if remaining == 0:
                rooms.pop(room_id, None)

        print(f"❌ Client left room '{room_id}' ({websocket.remote_address}); remaining={remaining}")

async def start_server():
    print("🚀 Room signaling server on ws://0.0.0.0:9000  (paths: /ws/<room>)")
    async with websockets.serve(
        signaling,
        "0.0.0.0",
        9000,
        process_request=process_request,
        max_size=2**23,
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(start_server())