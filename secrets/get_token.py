import asyncio
import websockets
import json
import pathlib

# Определяем директорию, где лежит сам скрипт
BASE_DIR = pathlib.Path(__file__).parent.resolve()
ICON_PATH = BASE_DIR / "icon_base64.txt"
TOKEN_PATH = BASE_DIR / "token.txt"


async def get_token():
    if not ICON_PATH.exists():
        print(f"❌ Ошибка: Файл {ICON_PATH} не найден!")
        return

    with open(ICON_PATH, "r", encoding="utf-8") as f:
        icon_b64 = f.read().strip()

    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "get_token",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": "VTracker API",
            "pluginDeveloper": "Myu",
            "pluginIcon": icon_b64,
        },
    }

    uri = "ws://127.0.0.1:8001"
    print("⏳ Подключение к VTube Studio...")

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(request))
            response_str = await websocket.recv()
            response = json.loads(response_str)

            if "data" in response and "authenticationToken" in response["data"]:
                token = response["data"]["authenticationToken"]

                # Записываем токен в файл
                with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                    f.write(token)

                print("✅ Токен успешно получен и сохранен!")
                print(f"📁 Путь: {TOKEN_PATH}")
            else:
                print(
                    "❌ Не удалось получить токен. Возможно, вы отклонили запрос в VTS."
                )
                print(f"Ответ сервера: {response_str}")

    except websockets.exceptions.ConnectionRefused:
        print("❌ Ошибка: VTube Studio не запущен или API отключен (порт 8001).")
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(get_token())
