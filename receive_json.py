import asyncio
import websockets
import json
import ssl

async def receive_messages():
    uri = "wss://signaling.ehb.be"   # Zelfde server als sender

    print("🔌 Verbinden met signaling server...")

    ssl_context = ssl.create_default_context()

    async with websockets.connect(uri,
        ssl=ssl_context,
        origin="https://signaling.ehb.be",
        compression=None,
        additional_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        },
    ) as websocket:
        print(f" Verbonden met signaling server ({uri})")

        while True:
            try:
                message = await websocket.recv()
                print("📩 Ruw bericht ontvangen:", message)

                # JSON parsen
                data = json.loads(message)

                print("📦 Geparsed JSON:")
                print("   Type :", data.get("type"))
                print("   From :", data.get("from"))
                print("   Data :", data.get("data"))
                print("-" * 40)

            except websockets.exceptions.ConnectionClosed:
                print("⚠ Verbinding met server verbroken.")
                break
            except json.JSONDecodeError:
                print("❌ Kon JSON niet parsen.")

asyncio.run(receive_messages())
