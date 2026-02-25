import asyncio
import ssl
import websockets

async def test():
    ssl_context = ssl.create_default_context()

    try:
        async with websockets.connect(
            "wss://signaling.ehb.be",
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
        ) as ws:
            print("✅ Connected!")
    except Exception as e:
        print("❌ Failed:", repr(e))

asyncio.run(test())