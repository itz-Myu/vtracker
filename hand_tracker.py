# hand_tracker.py
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import utils
from gestures import recognize_gesture

class HandTracker:
    def __init__(self, model_path=config.HAND_MODEL_PATH):
        self.model_path = model_path
        self.landmarker = None
        self._initialize()
        # Данные для сглаживания углов рук
        self.smooth_angle_x_left = 0.0
        self.smooth_angle_z_left = 0.0
        self.smooth_angle_x_right = 0.0
        self.smooth_angle_z_right = 0.0
        # Переменные для хранения состояния жестов
        self.last_gesture = None
        
    def _initialize(self):
        base = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
    
    def process(self, frame_rgb, frame_counter, calibration):
        """
        Обрабатывает кадр, возвращает словарь параметров рук, распознанный жест и список landmarks.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, frame_counter)

        hand_params = {}
        gesture = None
        finger_open_left = None
        finger_open_right = None
        landmarks_left = None
        landmarks_right = None
        landmarks_list = []  # <--- инициализация здесь

        if result and result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks_list.append(hand_landmarks)  # <--- добавляем все руки
                handedness = result.handedness[idx][0].category_name
                is_left = (handedness == "Left")

                mcp_ids = [2, 5, 9, 13, 17]
                tip_ids = [4, 8, 12, 16, 20]

                # Позиция руки
                wrist = hand_landmarks[0]
                pos_x = (wrist.x - 0.5) * 20.0
                pos_y = -(wrist.y - 0.5) * 20.0
                if not is_left:
                    pos_x = -pos_x
                pos_x = utils.clamp(pos_x, config.HAND_POSITION_X_MIN, config.HAND_POSITION_X_MAX)
                pos_y = utils.clamp(pos_y, config.HAND_POSITION_Y_MIN, config.HAND_POSITION_Y_MAX)

                # Пальцы
                finger_open = []
                for i in range(5):
                    tip = hand_landmarks[tip_ids[i]]
                    mcp = hand_landmarks[mcp_ids[i]]
                    diff_y = mcp.y - tip.y
                    max_diff = 0.1 if i == 0 else 0.08
                    val = max(0.0, min(1.0, diff_y / max_diff))
                    finger_open.append(val)
                finger_open[0] = 1.0 - finger_open[0]
                hand_open = sum(finger_open) / 5.0

                # Углы (только левая)
                if is_left:
                    index_base = hand_landmarks[5]
                    pinky_base = hand_landmarks[17]
                    dx = index_base.x - pinky_base.x
                    dy = index_base.y - pinky_base.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 0.001:
                        dist = 0.001
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    angle_x_raw = math.asin(max(-1.0, min(1.0, dx_norm))) * 180.0 / math.pi
                    angle_x_raw = utils.clamp(angle_x_raw, -config.HAND_ANGLE_LIMIT, config.HAND_ANGLE_LIMIT)
                    angle_z_raw = 0.0

                    if calibration.hand_ready:
                        offset = calibration.get_hand_offset("left")
                        angle_x_cal = angle_x_raw - offset["AngleX"]
                        angle_z_cal = angle_z_raw - offset["AngleZ"]
                        angle_x_cal = -angle_x_cal
                        self.smooth_angle_x_left = utils.smooth_value(self.smooth_angle_x_left, angle_x_cal, config.HAND_SMOOTHING)
                        self.smooth_angle_z_left = utils.smooth_value(self.smooth_angle_z_left, angle_z_cal, config.HAND_SMOOTHING)
                        self.smooth_angle_x_left = utils.dead_zone(self.smooth_angle_x_left, config.DEAD_ZONE)
                        self.smooth_angle_z_left = utils.dead_zone(self.smooth_angle_z_left, config.DEAD_ZONE)
                        hand_params["HandLeftAngleX"] = self.smooth_angle_x_left
                        hand_params["HandLeftAngleZ"] = self.smooth_angle_z_left

                    finger_open_left = finger_open
                    landmarks_left = hand_landmarks
                else:
                    # можно добавить углы для правой руки позже
                    finger_open_right = finger_open
                    landmarks_right = hand_landmarks

                # Общие параметры
                if is_left:
                    hand_params["HandLeftFound"] = 1.0
                    hand_params["HandLeftPositionX"] = pos_x
                    hand_params["HandLeftPositionY"] = pos_y
                    hand_params["HandLeftPositionZ"] = 0.0
                    hand_params["HandLeftOpen"] = hand_open
                    hand_params["HandLeftFinger_1_Thumb"] = finger_open[0]
                    hand_params["HandLeftFinger_2_Index"] = finger_open[1]
                    hand_params["HandLeftFinger_3_Middle"] = finger_open[2]
                    hand_params["HandLeftFinger_4_Ring"] = finger_open[3]
                    hand_params["HandLeftFinger_5_Pinky"] = finger_open[4]
                else:
                    hand_params["HandRightFound"] = 1.0
                    hand_params["HandRightPositionX"] = pos_x
                    hand_params["HandRightPositionY"] = pos_y
                    hand_params["HandRightPositionZ"] = 0.0
                    hand_params["HandRightOpen"] = hand_open
                    hand_params["HandRightFinger_1_Thumb"] = finger_open[0]
                    hand_params["HandRightFinger_2_Index"] = finger_open[1]
                    hand_params["HandRightFinger_3_Middle"] = finger_open[2]
                    hand_params["HandRightFinger_4_Ring"] = finger_open[3]
                    hand_params["HandRightFinger_5_Pinky"] = finger_open[4]

        # Распознаём жест (поддерживаем обе руки)
        if finger_open_left is not None and landmarks_left is not None:
            if finger_open_right is not None and landmarks_right is not None:
                gesture = recognize_gesture(finger_open_left, landmarks_left, finger_open_right, landmarks_right)
            else:
                gesture = recognize_gesture(finger_open_left, landmarks_left)
        elif finger_open_right is not None and landmarks_right is not None:
            # Распознаём жест на правой руке (передаём как левую)
            gesture = recognize_gesture(finger_open_right, landmarks_right)

        # Сбрасываем флаги, если руки не найдены
        if "HandLeftFound" not in hand_params:
            hand_params["HandLeftFound"] = 0.0
        if "HandRightFound" not in hand_params:
            hand_params["HandRightFound"] = 0.0

        return hand_params, gesture, landmarks_list
    
    def close(self):
        if self.landmarker:
            self.landmarker.close()
