import asyncio
import websockets
import json

async def receive_messages():
    uri = "ws://10.147.6.196:9000"   # Zelfde server als sender

    print("🔌 Verbinden met signaling server...")

    async with websockets.connect(uri) as websocket:
        print("✅ Verbonden! Wachten op berichten...\n")

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
