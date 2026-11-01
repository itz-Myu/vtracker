# face_tracker.py
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import numpy as np
import config
import utils


class FaceTracker:
    def __init__(self, model_path=config.FACE_MODEL_PATH):
        self.model_path = model_path
        self.landmarker = None
        self._initialize()
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0

        # Переменные для сглаживания мимики
        self.smooth_brows = 0.0
        self.smooth_eye_open_l = 1.0
        self.smooth_eye_open_r = 1.0
        self.smooth_eye_x = 0.0
        self.smooth_eye_y = 0.0

        # Сглаживание для Babble
        self.smooth_mouth_open = 0.0
        self.smooth_mouth_smile = 0.0
        self.smooth_tongue = 0.0
        self.smooth_cheek = 0.0

        self.calib_frames = []

        # --- Инициализация Babble ONNX ---
        babble_model_path = "tasks/model.onnx"  # или путь к файлу
        self.babble_session = ort.InferenceSession(
            babble_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Список выходных параметров Babble (стандартный для 45 значений)
        self.babble_blendshapes = [
            "JawForward",
            "JawLeft",
            "JawRight",
            "JawOpen",
            "MouthClose",
            "MouthFunnel",
            "MouthPucker",
            "MouthLeft",
            "MouthRight",
            "MouthSmileLeft",
            "MouthSmileRight",
            "MouthFrownLeft",
            "MouthFrownRight",
            "MouthDimpleLeft",
            "MouthDimpleRight",
            "MouthStretchLeft",
            "MouthStretchRight",
            "MouthRollLower",
            "MouthRollUpper",
            "MouthShrugLower",
            "MouthShrugUpper",
            "MouthPressLeft",
            "MouthPressRight",
            "MouthLowerDownLeft",
            "MouthLowerDownRight",
            "MouthUpperUpLeft",
            "MouthUpperUpRight",
            "CheekPuff",
            "CheekSquintLeft",
            "CheekSquintRight",
            "NoseSneerLeft",
            "NoseSneerRight",
            "TongueOut",
            "CheekSuck",
            "MouthRollIn",
            "MouthShrug",
            "MouthPress",
            "MouthUpperUp",
            "MouthLowerDown",
            "MouthUpperRight",
            "MouthUpperLeft",
            "MouthLowerRight",
            "MouthLowerLeft",
            "MouthUpperMid",
            "MouthLowerMid",
        ]

    def _initialize(self):
        base = python.BaseOptions(
            model_asset_path=self.model_path, delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_rgb, frame_counter, calibration):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, frame_counter)

        face_params = {}
        landmarks = None

        if result and result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            landmarks = face_landmarks
            face_params["faceFound"] = True

            # --- Углы из матрицы ---
            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]
                r00, r01, r02 = matrix[0][0], matrix[0][1], matrix[0][2]
                r10, r11, r12 = matrix[1][0], matrix[1][1], matrix[1][2]
                r20, r21, r22 = matrix[2][0], matrix[2][1], matrix[2][2]

                roll = math.atan2(-r10, r00)  # Z
                pitch = math.atan2(-r21, r22)  # X
                yaw = math.asin(r20)  # Y

                pitch_deg = pitch * 180 / math.pi
                yaw_deg = yaw * 180 / math.pi
                roll_deg = roll * 180 / math.pi
            else:
                pitch_deg = yaw_deg = roll_deg = 0.0

            # --- Калибровка ---
            if not calibration.face_ready:
                self.calib_frames.append((pitch_deg, yaw_deg, roll_deg))
                if len(self.calib_frames) >= config.CALIBRATION_FRAMES:
                    avg_pitch = sum(f[0] for f in self.calib_frames) / len(
                        self.calib_frames
                    )
                    avg_yaw = sum(f[1] for f in self.calib_frames) / len(
                        self.calib_frames
                    )
                    avg_roll = sum(f[2] for f in self.calib_frames) / len(
                        self.calib_frames
                    )
                    calibration.set_face_offset(avg_yaw, avg_pitch, avg_roll)
                    print("🎯 Калибровка лица завершена!")
                    self.calib_frames = []
                pitch_cal = pitch_deg
                yaw_cal = yaw_deg
                roll_cal = roll_deg
            else:
                offset = calibration.get_face_offset()
                pitch_cal = pitch_deg - offset["FaceAngleY"]
                yaw_cal = yaw_deg - offset["FaceAngleX"]
                roll_cal = roll_deg - offset["FaceAngleZ"]

            # --- Сглаживание углов ---
            self.smooth_pitch = utils.smooth_value(
                self.smooth_pitch, pitch_cal, config.FACE_SMOOTHING
            )
            self.smooth_yaw = utils.smooth_value(
                self.smooth_yaw, yaw_cal, config.FACE_SMOOTHING
            )
            self.smooth_roll = utils.smooth_value(
                self.smooth_roll, roll_cal, config.FACE_SMOOTHING
            )

            # --- Маппинг осей ---
            angle_values = {
                "pitch": self.smooth_pitch,
                "yaw": self.smooth_yaw,
                "roll": self.smooth_roll,
            }
            for param, source in config.FACE_AXIS_MAP.items():
                limit = (
                    config.FACE_ANGLE_X_LIMIT
                    if param in ("FaceAngleX", "FaceAngleY")
                    else config.FACE_ANGLE_Z_LIMIT
                )
                val = angle_values[source]
                if config.FACE_AXIS_INVERT.get(param, False):
                    val = -val
                face_params[param] = utils.clamp(val, -limit, limit)

            # --- Вспомогательная функция для моргания ---
            def map_eye_blink(blink_score):
                if blink_score < 0.2:
                    return 1.0
                elif blink_score > 0.6:
                    return 0.0
                else:
                    return 1.0 - (blink_score - 0.2) / 0.4

            # Переменные для сбора данных из блендшейпов
            brow_up = 0.0
            brow_down = 0.0
            look_in_left, look_out_left = 0.0, 0.0
            look_up_left, look_down_left = 0.0, 0.0
            look_in_right, look_out_right = 0.0, 0.0
            look_up_right, look_down_right = 0.0, 0.0
            eye_blink_l, eye_blink_r = 0.0, 0.0

            # --- ЕДИНЫЙ ЦИКЛ БЛЕНДШЕЙПОВ ---
            if result.face_blendshapes:
                for b in result.face_blendshapes[0]:
                    name = b.category_name
                    score = b.score

                    if name == "eyeBlinkLeft":
                        eye_blink_l = score
                    elif name == "eyeBlinkRight":
                        eye_blink_r = score
                    elif name == "browInnerUp":
                        brow_up = score
                    elif name in ("browDownLeft", "browDownRight"):
                        brow_down = max(brow_down, score)
                    elif name == "noseSneerLeft" or name == "noseSneerRight":
                        face_params["FaceAngry"] = max(
                            face_params.get("FaceAngry", 0), score
                        )

                    # Сбор взгляда
                    elif name == "eyeLookInLeft":
                        look_in_left = score
                    elif name == "eyeLookOutLeft":
                        look_out_left = score
                    elif name == "eyeLookUpLeft":
                        look_up_left = score
                    elif name == "eyeLookDownLeft":
                        look_down_left = score
                    elif name == "eyeLookInRight":
                        look_in_right = score
                    elif name == "eyeLookOutRight":
                        look_out_right = score
                    elif name == "eyeLookUpRight":
                        look_up_right = score
                    elif name == "eyeLookDownRight":
                        look_down_right = score

                # Усредняем улыбку
                if "MouthSmileLeft" in face_params and "MouthSmileRight" in face_params:
                    face_params["MouthSmile"] = (
                        face_params["MouthSmileLeft"] + face_params["MouthSmileRight"]
                    ) / 2
                    del face_params["MouthSmileLeft"]
                    del face_params["MouthSmileRight"]

            # --- Обработка рта и щёк через Project Babble (ONNX) ---
            h, w = frame_rgb.shape[:2]

            mouth_xs = [
                face_landmarks[61].x * w,
                face_landmarks[291].x * w,
                face_landmarks[0].x * w,
                face_landmarks[17].x * w,
            ]
            mouth_ys = [
                face_landmarks[61].y * h,
                face_landmarks[291].y * h,
                face_landmarks[0].y * h,
                face_landmarks[17].y * h,
            ]

            x1, x2 = min(mouth_xs), max(mouth_xs)
            y1, y2 = min(mouth_ys), max(mouth_ys)

            box_w, box_h = x2 - x1, y2 - y1
            # УВЕЛИЧИЛИ ОТСТУП С 1.8 ДО 2.2 (чтобы захватить губы при "Ы" и "О")
            side = max(box_w, box_h) * 2.2

            cx, cy = x1 + box_w / 2, y1 + box_h / 2
            x1 = int(max(0, cx - side / 2))
            x2 = int(min(w, cx + side / 2))
            y1 = int(max(0, cy - side / 2))
            y2 = int(min(h, cy + side / 2))

            mouth_roi = frame_rgb[y1:y2, x1:x2]
            if mouth_roi.size > 0:
                mouth_roi = cv2.resize(mouth_roi, (224, 224))
                gray = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)
                input_tensor = gray.astype(np.float32) / 255.0
                input_tensor = np.expand_dims(input_tensor, axis=0)
                input_tensor = np.expand_dims(input_tensor, axis=0)

                input_name = self.babble_session.get_inputs()[0].name
                babble_output = self.babble_session.run(
                    None, {input_name: input_tensor}
                )[0][0]

                babble_dict = {
                    name: float(babble_output[i])
                    for i, name in enumerate(self.babble_blendshapes)
                    if i < len(babble_output)
                }

                # 1. Считаем открытие рта (JawOpen)
                raw_jaw = babble_dict.get("JawOpen", 0.0)
                # УВЕЛИЧИЛИ МНОЖИТЕЛЬ ДО 4.5 (чтобы "Ы" и "О" открывали рот до конца)
                target_mouth = min(1.0, max(0.0, raw_jaw * 4.5))

                # 2. Считаем улыбку
                raw_smile = max(
                    babble_dict.get("MouthSmileLeft", 0.0),
                    babble_dict.get("MouthSmileRight", 0.0),
                )
                if raw_smile < 0.06:
                    target_smile = 0.0
                else:
                    target_smile = min(1.0, (raw_smile - 0.06) * 8.0)

                # 3. Гасим улыбку, если рот широко открыт
                if target_mouth > 0.4:
                    target_smile = target_smile * max(
                        0.0, 1.0 - (target_mouth - 0.4) * 2.5
                    )

                # 4. МЕДИАННЫЙ ФИЛЬТР ДЛЯ ЧЕЛЮСТИ (лечит мерцание)
                if not hasattr(self, "jaw_history"):
                    self.jaw_history = []
                self.jaw_history.append(target_mouth)
                if len(self.jaw_history) > 3:
                    self.jaw_history.pop(0)
                # Берем среднее значение из 3 последних кадров, чтобы сгладить провалы
                stable_mouth = sum(self.jaw_history) / len(self.jaw_history)

                target_tongue = babble_dict.get("TongueOut", 0.0)
                target_cheek = babble_dict.get("CheekPuff", 0.0)

                self.smooth_mouth_open = utils.smooth_value(
                    self.smooth_mouth_open, stable_mouth, 0.5
                )
                self.smooth_mouth_smile = utils.smooth_value(
                    self.smooth_mouth_smile, target_smile, 0.2
                )

                self.smooth_tongue = utils.smooth_value(
                    self.smooth_tongue, target_tongue, 0.3
                )
                self.smooth_cheek = utils.smooth_value(
                    self.smooth_cheek, target_cheek, 0.3
                )

                face_params["MouthOpen"] = self.smooth_mouth_open
                face_params["MouthSmile"] = self.smooth_mouth_smile
                face_params["TongueOut"] = self.smooth_tongue
                face_params["CheekPuff"] = self.smooth_cheek

                print(
                    f"BABBLE -> VTS: Open={face_params['MouthOpen']:.2f}, Smile={face_params['MouthSmile']:.2f}, JawRaw={raw_jaw:.2f}"
                )

            # --- Применяем сглаживание и расчеты ---

            # 1. Брови
            brows_value = utils.clamp((brow_up * 1.2) - (brow_down * 0.6), -0.3, 1.2)
            self.smooth_brows = utils.smooth_value(self.smooth_brows, brows_value, 0.15)
            face_params["Brows"] = self.smooth_brows

            # 2. Моргание (сглаживаем, чтобы не дёргалось)
            target_eye_l = map_eye_blink(eye_blink_l)
            target_eye_r = map_eye_blink(eye_blink_r)
            self.smooth_eye_open_l = utils.smooth_value(
                self.smooth_eye_open_l, target_eye_l, 1
            )
            self.smooth_eye_open_r = utils.smooth_value(
                self.smooth_eye_open_r, target_eye_r, 1
            )
            face_params["EyeOpenLeft"] = self.smooth_eye_open_l
            face_params["EyeOpenRight"] = self.smooth_eye_open_r

            # 3. Взгляд
            horizontal = (look_out_left + look_in_right) - (
                look_in_left + look_out_right
            )
            vertical = (look_up_left + look_up_right) - (
                look_down_left + look_down_right
            )

            horizontal = utils.clamp(horizontal, -1.0, 1.0)
            vertical = utils.clamp(vertical, -1.0, 1.0)

            # Сглаживаем взгляд
            self.smooth_eye_x = utils.smooth_value(self.smooth_eye_x, horizontal, 0.5)
            self.smooth_eye_y = utils.smooth_value(self.smooth_eye_y, vertical, 0.5)

            face_params["EyeLeftX"] = self.smooth_eye_x
            face_params["EyeRightX"] = self.smooth_eye_x
            face_params["EyeLeftY"] = self.smooth_eye_y
            face_params["EyeRightY"] = self.smooth_eye_y

        else:
            face_params["faceFound"] = False

        return face_params, landmarks

    def close(self):
        if self.landmarker:
            self.landmarker.close()
