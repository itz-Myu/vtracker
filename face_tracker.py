# face_tracker.py
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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
        self.smooth_brow_left = 0.0
        self.smooth_brow_right = 0.0
        self.smooth_eye_open_l = 1.0
        self.smooth_eye_open_r = 1.0
        self.smooth_eye_x = 0.0
        self.smooth_eye_y = 0.0
        # Для CheekPuff
        self.smooth_cheek_puff = 0.0
        self.base_cheek_dist = None

        self.calib_frames = []

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
            brow_down_left = 0.0
            brow_down_right = 0.0
            brow_outer_up_left = 0.0
            brow_outer_up_right = 0.0
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
                    elif name == "jawOpen":
                        face_params["MouthOpen"] = score
                    elif name == "mouthSmileLeft":
                        face_params["MouthSmileLeft"] = score
                    elif name == "mouthSmileRight":
                        face_params["MouthSmileRight"] = score
                    elif name == "mouthFunnel":
                        face_params["MouthFunnel"] = score
                    elif name == "mouthShrugLower":
                        face_params["MouthShrugLower"] = score
                    elif name == "tongueOut":
                        face_params["TongueOut"] = score
                    elif name == "browInnerUp":
                        brow_up = score
                    elif name == "browDownLeft":
                        brow_down_left = score
                    elif name == "browDownRight":
                        brow_down_right = score
                    elif name == "browOuterUpLeft":
                        brow_outer_up_left = score
                    elif name == "browOuterUpRight":
                        brow_outer_up_right = score
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

            # --- CheekPuff (3D Геометрия + защита от наклонов) ---
            geo_cheek_dist_avg = 0.0
            if face_landmarks:
                nose_tip = face_landmarks[4]
                left_cheek = face_landmarks[205]
                right_cheek = face_landmarks[425]

                # 3D Евклидово расстояние (спасает от сжатия перспективы)
                d_left = math.sqrt(
                    (nose_tip.x - left_cheek.x)**2 + 
                    (nose_tip.y - left_cheek.y)**2 + 
                    (nose_tip.z - left_cheek.z)**2
                )
                d_right = math.sqrt(
                    (nose_tip.x - right_cheek.x)**2 + 
                    (nose_tip.y - right_cheek.y)**2 + 
                    (nose_tip.z - right_cheek.z)**2
                )
                geo_cheek_dist_avg = (d_left + d_right) / 2.0

                # Инициализация базы (только когда голова ровно)
                if self.base_cheek_dist is None and abs(self.smooth_pitch) < 15.0 and abs(self.smooth_yaw) < 15.0:
                    self.base_cheek_dist = geo_cheek_dist_avg

            cheek_puff_value = 0.0
            rel_change = 0.0

            if self.base_cheek_dist and self.base_cheek_dist > 0:
                rel_change = (geo_cheek_dist_avg - self.base_cheek_dist) / self.base_cheek_dist

                CHEEK_PUFF_THRESHOLD = 0.03  # 3% мертвая зона
                CHEEK_PUFF_MAX = 0.10        # 10% полное надувание

                if rel_change > CHEEK_PUFF_THRESHOLD:
                    cheek_puff_value = utils.clamp(
                        (rel_change - CHEEK_PUFF_THRESHOLD) / (CHEEK_PUFF_MAX - CHEEK_PUFF_THRESHOLD), 
                        0.0, 1.0
                    )

            # --- ПЛАВНАЯ ЗАЩИТА ОТ НАКЛОНОВ ---
            pitch_factor = max(0.0, min(1.0, (25.0 - abs(self.smooth_pitch)) / 10.0))
            yaw_factor = max(0.0, min(1.0, (25.0 - abs(self.smooth_yaw)) / 10.0))

            # Умножаем значение на эти коэффициенты
            cheek_puff_value *= (pitch_factor * yaw_factor)

            # Фильтр по рту (если открыт - не надуваем)
            if face_params.get("MouthOpen", 0.0) > 0.4:
                cheek_puff_value = 0.0

            # Сглаживание
            self.smooth_cheek_puff = utils.smooth_value(self.smooth_cheek_puff, cheek_puff_value, 0.2)
            face_params["CheekPuff"] = self.smooth_cheek_puff

            # --- Применяем сглаживание и расчеты ---

            # 1. Брови (раздельно левая и правая)
            # Внутренняя часть поднимает обе брови, внешняя и опускание - индивидуально
            left_brow_raw = (brow_up * 1.0) + (brow_outer_up_left * 0.8) - (brow_down_left * 0.6)
            right_brow_raw = (brow_up * 1.0) + (brow_outer_up_right * 0.8) - (brow_down_right * 0.6)

            # Клипаем в диапазон 0..1 (VTS ждёт 0..1 для BrowLeftY/RightY)
            target_brow_l = utils.clamp(left_brow_raw, 0.0, 1.0)
            target_brow_r = utils.clamp(right_brow_raw, 0.0, 1.0)

            self.smooth_brow_left = utils.smooth_value(self.smooth_brow_left, target_brow_l, 0.15)
            self.smooth_brow_right = utils.smooth_value(self.smooth_brow_right, target_brow_r, 0.15)

            face_params["BrowLeftY"] = self.smooth_brow_left
            face_params["BrowRightY"] = self.smooth_brow_right

            # Оставляем общий Brows для обратной совместимости 
            # (если в других моделях он используется)
            self.smooth_brows = (self.smooth_brow_left + self.smooth_brow_right) / 2.0
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
