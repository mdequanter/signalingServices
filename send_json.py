import asyncio
import websockets
import json
import ssl

SIGNALING_SERVER = "wss://signaling.ehb.be"

async def send_message():
    #uri = "ws://192.168.0.73:9000"   # Pas dit aan als server elders draait
    uri = SIGNALING_SERVER

    # JSON bericht dat je wil sturen
    message = {
        "type": "topic",
        "from": "client1",
        "data": {
            "name": "topic1",
            "value": "Any value"
        }
    }

    

    ssl_context = ssl.create_default_context()
    
    async with websockets.connect(SIGNALING_SERVER,
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
        print(f" Verbonden met signaling server ({SIGNALING_SERVER})")

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
