import asyncio
import websockets
import json

async def send_message():
    uri = "ws://localhost:9000"   # Pas dit aan als server elders draait

    # JSON bericht dat je wil sturen
    message = {
        "type": "offer",
        "from": "client1",
        "data": {
            "sdp": "voorbeeld SDP string"
        }
    }

    print("🔌 Verbinden met signaling server...")
    
    async with websockets.connect(uri) as websocket:
        print("✅ Verbonden!")

        # Converteer naar JSON-string
        json_message = json.dumps(message)

        print("📤 Bericht versturen:", json_message)

        # Verstuur bericht
        await websocket.send(json_message)

        print("✅ Bericht verzonden!")

        # Optioneel: wachten op antwoord
        # response = await websocket.recv()
        # print("📩 Antwoord ontvangen:", response)

asyncio.run(send_message())
