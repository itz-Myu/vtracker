# vts_client.py
import asyncio
import json
import websockets
import time

class VTSClient:
    def __init__(self, ip, port, plugin_name, plugin_developer, plugin_token):
        self.ip = ip
        self.port = port
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.plugin_token = plugin_token
        self.websocket = None
        self.connected = False
        self.expressions = {}
        self.last_toggle_time = 0

    async def connect(self):
        uri = f"ws://{self.ip}:{self.port}"
        try:
            self.websocket = await websockets.connect(uri, ping_interval=30, ping_timeout=30)
            await self._authenticate()
            await self._get_expressions()
            self.connected = True
            print("✅ Подключено к VTube Studio")
        except Exception as e:
            print(f"❌ Ошибка подключения к VTS: {e}")
            self.connected = False
            raise

    async def _authenticate(self):
        auth_msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth1",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "authenticationToken": self.plugin_token
            }
        }
        await self.websocket.send(json.dumps(auth_msg))
        response = await self.websocket.recv()
        if '"authenticated":true' not in response:
            raise Exception(f"Ошибка аутентификации: {response}")
        print("✅ Аутентификация успешна")

    async def _get_expressions(self):
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "get_expr",
            "messageType": "ExpressionStateRequest"
        }
        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        try:
            data = json.loads(response)
            if "data" in data and "expressions" in data["data"]:
                for expr in data["data"]["expressions"]:
                    self.expressions[expr["name"]] = expr["file"]
                print(f"📋 Получено {len(self.expressions)} выражений: {list(self.expressions.keys())}")
        except:
            print("⚠️ Не удалось получить список выражений")

    async def send_parameters(self, param_list):
        if not self.connected:
            print("⚠️ Не отправляем параметры, так как VTS не подключён")
            return
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "params",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "faceFound": True,
                "parameterValues": param_list
            }
        }
        try:
            await self.websocket.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed as e:
            print(f"⚠️ Соединение закрыто при отправке параметров: {e}")
            self.connected = False
        except Exception as e:
            print(f"⚠️ Ошибка отправки параметров: {e}")
            self.connected = False

    async def toggle_expression(self, alias, active):
        if not self.connected:
            print(f"⚠️ Не отправляем выражение '{alias}', так как VTS не подключён")
            return
        if alias not in self.expressions:
            print(f"⚠️ Выражение '{alias}' не найдено в модели (доступны: {list(self.expressions.keys())})")
            return
        # Ограничиваем частоту отправки (не чаще 1 раза в 0.5 секунды)
        now = time.time()
        if now - self.last_toggle_time < 0.5:
            return
        self.last_toggle_time = now

        expr_file = self.expressions[alias]
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"toggle_{alias}",
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": expr_file,
                "active": active
            }
        }
        try:
            await self.websocket.send(json.dumps(request))
        except websockets.exceptions.ConnectionClosed as e:
            print(f"⚠️ Соединение закрыто при отправке выражения: {e}")
            self.connected = False
        except Exception as e:
            print(f"⚠️ Ошибка отправки выражения: {e}")
            self.connected = False

    async def close(self):
        if self.websocket:
            await self.websocket.close()
