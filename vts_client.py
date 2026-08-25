# vts_client.py
import asyncio
import json
import websockets

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
        # Парсим и сохраняем выражения
        try:
            data = json.loads(response)
            if "data" in data and "expressions" in data["data"]:
                for expr in data["data"]["expressions"]:
                    self.expressions[expr["name"]] = expr["file"]
            print(f"📋 Получено {len(self.expressions)} выражений")
        except:
            print("⚠️ Не удалось получить список выражений")
    
    async def send_parameters(self, param_list):
        """param_list — список словарей [{"id": "FaceAngleX", "value": 0.0}, ...]"""
        if not self.connected:
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
            try:
                await asyncio.wait_for(self.websocket.recv(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"⚠️ Ошибка отправки параметров: {e}")
            self.connected = False

    async def toggle_expression(self, gesture_name, active):
        if gesture_name not in self.expressions:
            print(f"⚠️ Выражение '{gesture_name}' не найдено")
            return
        expr_file = self.expressions[gesture_name]
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "toggle_expr",
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressionFile": expr_file,
                "active": active
            }
        }
        try:
            await self.websocket.send(json.dumps(request))
        except Exception as e:
            print(f"⚠️ Ошибка активации выражения: {e}")
    
    async def close(self):
        if self.websocket:
            await self.websocket.close()
