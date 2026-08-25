# config.py
import os

def read_secret(filename):
    with open(os.path.join("secrets", filename), "r") as f:
        return f.read().strip()

VTS_IP = "127.0.0.1"
VTS_PORT = 8001

PLUGIN_NAME = "API VTracker"
PLUGIN_DEVELOPER = "Myu"
PLUGIN_TOKEN = read_secret("token.txt")
PLUGIN_ICON = read_secret("icon_base64.txt")

CAMERA_ID = 0
WINDOW_NAME = "VT Tracker (MediaPipe)"
FACE_MODEL_PATH = "tasks/face_landmarker.task"
HAND_MODEL_PATH = "tasks/hand_landmarker.task"

# Настройки камеры
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 576
CAMERA_FPS = 30
CAMERA_FORMAT = 'MJPG'  # или 'YUYV'

# Калибровка
CALIB_FILE = "calibration.json"
CALIBRATION_FRAMES = 30
HAND_CALIBRATION_FRAMES = 30

# Сглаживание
FACE_SMOOTHING = 0.2
HAND_SMOOTHING = 0.05
ANGLE_SMOOTHING = 0.05

# Ограничения
FACE_ANGLE_X_LIMIT = 30
FACE_ANGLE_Y_LIMIT = 30
FACE_ANGLE_Z_LIMIT = 90
HAND_POSITION_X_MIN = 0
HAND_POSITION_X_MAX = 10
HAND_POSITION_Y_MIN = -10
HAND_POSITION_Y_MAX = 10
HAND_ANGLE_LIMIT = 90

# Мёртвая зона для углов
DEAD_ZONE = 2.0
