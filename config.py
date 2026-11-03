# config.py
import os

def read_secret(filename):
    with open(os.path.join("secrets", filename), "r") as f:
        return f.read().strip()

VTS_IP   = "127.0.0.1"
VTS_PORT = 8001

PLUGIN_NAME      = "API VTracker"
PLUGIN_DEVELOPER = "Myu"
PLUGIN_TOKEN     = read_secret("token.txt")
PLUGIN_ICON      = read_secret("icon_base64.txt")

CAMERA_ID       = 0
WINDOW_NAME     = "VTracker Visualization"
FACE_MODEL_PATH = "tasks/face_landmarker.task"
HAND_MODEL_PATH = "tasks/hand_landmarker.task"

# Настройки камеры
CAMERA_WIDTH  = 1024
CAMERA_HEIGHT = 576
CAMERA_FPS    = 30
CAMERA_FORMAT = "MJPG"  # 'YUYV'

# Калибровка
CALIB_FILE = "calibration.json"
CALIBRATION_FRAMES = 30
HAND_CALIBRATION_FRAMES = 30

# Настройки головы
FACE_AXIS_INVERT = {
    "FaceAngleX": False,
    "FaceAngleY": False,
    "FaceAngleZ": True,
}

FACE_AXIS_MAP = {
    "FaceAngleX": "yaw",
    "FaceAngleY": "pitch",
    "FaceAngleZ": "roll",
}

# --- Настройки жестов ---
GESTURE_THRESHOLD_EXTENDED = 0.6  # порог для разогнутых пальцев (0..1)
GESTURE_THRESHOLD_FOLDED   = 0.35  # порог для согнутых пальцев (0..1)
GESTURE_DISTANCE_THRESHOLD = 0.08  # порог расстояния для жеста OK (в нормализованных координатах)

## Индивидуальные пороги
GESTURE_THUMB_EXTENDED = 0.45
GESTURE_THUMB_FOLDED   = 0.35

## Общие алиасы
GESTURE_ALIAS = {
    "fist": "Fist",
    "open_palm": "Open",
    "peace": "Peace",
    "ok": "OK",
    "thumbs_up": "ThumbsUp",
    "middle_finger": "Middle",
    "point_up": "Point",
    "pinky_up": "Pinky",
    "horns": "Horns",
    "three_fingers": "Three",
    "four_fingers": "Four",
}

## Алиасы для левой руки
GESTURE_ALIAS_LEFT = {
    # "horns": "handL",
}

## Алиасы для правой руки
GESTURE_ALIAS_RIGHT = {
    # "horns": "handR",
}

## Алиасы для двухручных жестов
GESTURE_TWO_HAND_ALIAS = {
    "heart": "eye squint",
    "fingers_together": "boykisser face",
}

## Комбинации жестов
GESTURE_COMBINATIONS = {
    ("peace", "peace"): "eye blink mode",
    ("fist", "fist"):   "arms toggle",
}

# Сглаживание
FACE_SMOOTHING  = 0.4
HAND_SMOOTHING  = 0.05
ANGLE_SMOOTHING = 0.05
DEBOUNCE_FRAMES = 16

# Ограничения
FACE_ANGLE_X_LIMIT  = 30
FACE_ANGLE_Y_LIMIT  = 30
FACE_ANGLE_Z_LIMIT  = 90
HAND_POSITION_X_MIN = 0
HAND_POSITION_X_MAX = 10
HAND_POSITION_Y_MIN = -10
HAND_POSITION_Y_MAX = 10
HAND_ANGLE_LIMIT    = 90

# Мёртвая зона для углов
DEAD_ZONE = 2.0
