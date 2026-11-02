# get.py
import asyncio
import json
import websockets
import config


async def _request(ws, message_type, request_id, data=None):
    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": request_id,
        "messageType": message_type,
        "data": data or {},
    }
    await ws.send(json.dumps(msg))
    return json.loads(await ws.recv())


def _print_params(title, params):
    print(f"{title} ({len(params)}):")
    if not params:
        print("   (пусто)")
    for p in params:
        print(
            f"   • {p['name']:<32s} "
            f"value={p['value']:+.3f}  "
            f"range=[{p['min']:+.1f}, {p['max']:+.1f}]"
        )
    print()


async def main():
    uri = f"ws://{config.VTS_IP}:{config.VTS_PORT}"
    async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as ws:
        auth = await _request(
            ws,
            "AuthenticationRequest",
            "auth1",
            {
                "pluginName": config.PLUGIN_NAME,
                "pluginDeveloper": config.PLUGIN_DEVELOPER,
                "authenticationToken": config.PLUGIN_TOKEN,
            },
        )
        if not auth.get("data", {}).get("authenticated"):
            print("❌ Ошибка аутентификации:", auth)
            return
        print("✅ Аутентификация успешна\n")

        # --- Выражения (то, на что мапятся жесты) ---
        resp = await _request(ws, "ExpressionStateRequest", "expr_list")
        expressions = resp.get("data", {}).get("expressions", [])
        print(f"📋 Выражения в модели ({len(expressions)}):")
        for e in expressions:
            print(f"   • {e['name']:<24s} → {e['file']}")
        print()

        # --- Параметры ---
        resp = await _request(ws, "InputParameterListRequest", "param_list")
        data = resp.get("data", {})
        print(
            f"🎭 Модель: {data.get('modelName', '?')}  "
            f"(id={data.get('modelID', '?')[:8]}…)"
        )
        print(f"   loaded: {data.get('modelLoaded')}\n")

        _print_params("⚙️  Кастомные параметры", data.get("customParameters", []))

        # Группируем дефолтные параметры по префиксу
        groups = {
            "Face": [],
            "Mouth": [],
            "Eye": [],
            "Brow": [],
            "Voice": [],
            "Hand": [],
            "Mocopi": [],
            "Controller": [],
        }
        for p in data.get("defaultParameters", []):
            n = p["name"]
            if n.startswith("Controller"):
                groups["Controller"].append(p)
            elif n.startswith("Mocopi"):
                groups["Mocopi"].append(p)
            elif n.startswith(("Hand", "BothHands")):
                groups["Hand"].append(p)
            elif n.startswith("Voice"):
                groups["Voice"].append(p)
            elif n.startswith("Brow"):
                groups["Brow"].append(p)
            elif n.startswith("Eye"):
                groups["Eye"].append(p)
            elif n.startswith("Mouth"):
                groups["Mouth"].append(p)
            else:
                groups["Face"].append(p)

        for title, params in groups.items():
            if params:
                _print_params(f"🎛️ {title}", params)


if __name__ == "__main__":
    asyncio.run(main())
