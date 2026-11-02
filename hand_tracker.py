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
        # Сглаживание для левой руки
        self.smooth_angle_x_left = 0.0
        self.smooth_angle_z_left = 0.0
        # Сглаживание для правой руки
        self.smooth_angle_x_right = 0.0
        self.smooth_angle_z_right = 0.0
        # Стабилизация жестов
        self.gesture_history_left = []
        self.gesture_history_right = []
        # Калибровочные кадры
        self.calib_hand_frames_left = []
        self.calib_hand_frames_right = []
        self.last_gesture_left = None
        self.last_gesture_right = None

    def _initialize(self):
        base = python.BaseOptions(
            model_asset_path=self.model_path, delegate=python.BaseOptions.Delegate.GPU
        )
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, frame_counter)

        hand_params = {}
        gesture_left = None
        gesture_right = None
        finger_open_left = None
        finger_open_right = None
        landmarks_left = None
        landmarks_right = None
        landmarks_list = []

        if result and result.hand_landmarks:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks_list.append(hand_landmarks)
                handedness = result.handedness[idx][0].category_name
                is_left = handedness == "Left"

                mcp_ids = [2, 5, 9, 13, 17]
                tip_ids = [4, 8, 12, 16, 20]

                # Позиция руки
                wrist = hand_landmarks[0]
                pos_x = (wrist.x - 0.5) * 20.0
                pos_y = -(wrist.y - 0.5) * 20.0
                if not is_left:
                    pos_x = -pos_x
                pos_x = utils.clamp(
                    pos_x, config.HAND_POSITION_X_MIN, config.HAND_POSITION_X_MAX
                )
                pos_y = utils.clamp(
                    pos_y, config.HAND_POSITION_Y_MIN, config.HAND_POSITION_Y_MAX
                )

                # Пальцы
                finger_open = []

                # --- Большой палец (0) ---
                # Считаем расстояние от кончика большого до основания указательного.
                # Нормируем на размер ладони (запястье -> основание среднего).
                tip_thumb = hand_landmarks[4]
                index_mcp = hand_landmarks[5]
                wrist = hand_landmarks[0]
                middle_mcp = hand_landmarks[9]

                dist_thumb_index = math.sqrt((tip_thumb.x - index_mcp.x)**2 + (tip_thumb.y - index_mcp.y)**2)
                palm_size = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)

                if palm_size > 1e-6:
                    ratio = dist_thumb_index / palm_size
                    # При открытом ~ 1.2, при прижатом (ОК/кулак) ~ 0.4. 
                    # Шкала 0.4 -> 0.0, 1.0 -> 1.0
                    thumb_val = max(0.0, min(1.0, (ratio - 0.4) / 0.6))
                else:
                    thumb_val = 0.0
                finger_open.append(thumb_val)

                # --- Остальные пальцы (1-4) ---
                for i in range(1, 5):
                    mcp = hand_landmarks[mcp_ids[i]]
                    pip = hand_landmarks[mcp_ids[i] + 1]
                    tip = hand_landmarks[tip_ids[i]]

                    # ИСПОЛЬЗУЕМ ТОЛЬКО 2D (x, y), без z! Это уберет шум.
                    v1 = (pip.x - mcp.x, pip.y - mcp.y)
                    v2 = (tip.x - pip.x, tip.y - pip.y)

                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

                    if mag1 > 1e-6 and mag2 > 1e-6:
                        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                        angle = math.degrees(math.acos(cos_a))
                        # Палец редко сгибается ровно на 90 градусов (обычно ~110).
                        # Мапим 0 -> 1.0, 110 -> 0.0
                        val = max(0.0, min(1.0, 1.0 - (angle / 110.0)))
                    else:
                        val = 0.0
                    finger_open.append(val)

                hand_open = sum(finger_open) / 5.0

                # --- УГЛЫ И КАЛИБРОВКА ---
                index_base = hand_landmarks[5]
                pinky_base = hand_landmarks[17]
                dx = index_base.x - pinky_base.x
                dy = index_base.y - pinky_base.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 0.001:
                    dist = 0.001

                # ВОЗВРАЩАЕМ ASIN (он не делает скачков и не ломает сглаживание)
                angle_x_raw = math.asin(max(-1.0, min(1.0, dx / dist))) * 180.0 / math.pi
                angle_x_raw = utils.clamp(angle_x_raw, -config.HAND_ANGLE_LIMIT, config.HAND_ANGLE_LIMIT)
                angle_z_raw = 0.0  # пока не используем

                if is_left:
                    # --- КАЛИБРОВКА ЛЕВОЙ РУКИ ---
                    if not calibration.hand_left_ready:
                        self.calib_hand_frames_left.append((angle_x_raw, angle_z_raw))
                        if len(self.calib_hand_frames_left) >= config.HAND_CALIBRATION_FRAMES:
                            avg_x = sum(f[0] for f in self.calib_hand_frames_left) / len(self.calib_hand_frames_left)
                            avg_z = sum(f[1] for f in self.calib_hand_frames_left) / len(self.calib_hand_frames_left)
                            calibration.set_hand_offset("left", avg_x, avg_z)
                            print("🎯 Калибровка левой руки завершена!")
                            self.calib_hand_frames_left = []
                        hand_params["HandLeftAngleX"] = 0.0
                        hand_params["HandLeftAngleZ"] = 0.0
                    else:
                        offset = calibration.get_hand_offset("left")
                        angle_x_cal = angle_x_raw - offset["AngleX"]
                        angle_z_cal = angle_z_raw - offset["AngleZ"]

                        # ИНВЕРСИЯ ДЛЯ ЛЕВОЙ РУКИ (ты был прав, она зеркальна)
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
                    # --- КАЛИБРОВКА ПРАВОЙ РУКИ ---
                    if not calibration.hand_right_ready:
                        self.calib_hand_frames_right.append((angle_x_raw, angle_z_raw))
                        if len(self.calib_hand_frames_right) >= config.HAND_CALIBRATION_FRAMES:
                            avg_x = sum(f[0] for f in self.calib_hand_frames_right) / len(self.calib_hand_frames_right)
                            avg_z = sum(f[1] for f in self.calib_hand_frames_right) / len(self.calib_hand_frames_right)
                            calibration.set_hand_offset("right", avg_x, avg_z)
                            print("🎯 Калибровка правой руки завершена!")
                            self.calib_hand_frames_right = []
                        hand_params["HandRightAngleX"] = 0.0
                        hand_params["HandRightAngleZ"] = 0.0
                    else:
                        offset = calibration.get_hand_offset("right")
                        angle_x_cal = angle_x_raw - offset["AngleX"]
                        angle_z_cal = angle_z_raw - offset["AngleZ"]

                        # ДЛЯ ПРАВОЙ РУКИ ИНВЕРСИИ НЕТ (она эталонная)
                        angle_x_cal = angle_x_cal 

                        self.smooth_angle_x_right = utils.smooth_value(self.smooth_angle_x_right, angle_x_cal, config.HAND_SMOOTHING)
                        self.smooth_angle_z_right = utils.smooth_value(self.smooth_angle_z_right, angle_z_cal, config.HAND_SMOOTHING)
                        self.smooth_angle_x_right = utils.dead_zone(self.smooth_angle_x_right, config.DEAD_ZONE)
                        self.smooth_angle_z_right = utils.dead_zone(self.smooth_angle_z_right, config.DEAD_ZONE)
                        hand_params["HandRightAngleX"] = self.smooth_angle_x_right
                        hand_params["HandRightAngleZ"] = self.smooth_angle_z_right

                    finger_open_right = finger_open
                    landmarks_right = hand_landmarks

                # Общие параметры (позиция, открытие, пальцы)
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

        # --- РАСПОЗНАВАНИЕ ЖЕСТОВ ---
        # Проверяем двухручные жесты (если есть обе руки)
        two_hand_gesture = None
        if (
            finger_open_left is not None
            and landmarks_left is not None
            and finger_open_right is not None
            and landmarks_right is not None
        ):
            two_hand_gesture = recognize_gesture(
                finger_open_left, landmarks_left, finger_open_right, landmarks_right
            )

        if two_hand_gesture:
            # двухручный жест имеет приоритет
            gesture_left = two_hand_gesture
            gesture_right = two_hand_gesture
        else:
            # распознаём каждую руку отдельно
            if finger_open_left is not None and landmarks_left is not None:
                gesture_left = recognize_gesture(finger_open_left, landmarks_left)
            if finger_open_right is not None and landmarks_right is not None:
                gesture_right = recognize_gesture(finger_open_right, landmarks_right)

        # Сбрасываем флаги, если руки не найдены
        if "HandLeftFound" not in hand_params:
            hand_params["HandLeftFound"] = 0.0
        if "HandRightFound" not in hand_params:
            hand_params["HandRightFound"] = 0.0

        # --- Стабилизация жестов (скользящее окно 5 кадров) ---
        def stabilize(gesture, history):
            history.append(gesture)
            if len(history) > 5:
                history.pop(0)

            # Ищем самый частый жест в окне
            valid_gestures = [g for g in history if g is not None]
            if valid_gestures:
                most_common = max(set(valid_gestures), key=valid_gestures.count)
                # Если самый частый жест встретился хотя бы 4 раза из 5
                if valid_gestures.count(most_common) >= 4:
                    return most_common
            return None

        stable_left = stabilize(gesture_left, self.gesture_history_left)
        stable_right = stabilize(gesture_right, self.gesture_history_right)

        # Сброс истории, если рука пропала
        if "HandLeftFound" not in hand_params:
            hand_params["HandLeftFound"] = 0.0
            self.gesture_history_left.clear()
        if "HandRightFound" not in hand_params:
            hand_params["HandRightFound"] = 0.0
            self.gesture_history_right.clear()

        return hand_params, stable_left, stable_right, landmarks_list

    def close(self):
        if self.landmarker:
            self.landmarker.close()
