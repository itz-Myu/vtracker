import asyncio
import websockets
import json


async def get_token():
    with open("icon_base64.txt", "r") as f:
        icon_b64 = f.read().strip()

    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "get_token",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": "API VTracker",
            "pluginDeveloper": "Myu",
            "pluginIcon": icon_b64,
        },
    }

    uri = "ws://127.0.0.1:8001"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        print("Ответ от VTube Studio:")
        print(response)


asyncio.run(get_token())
