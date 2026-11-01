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
        self.smooth_eye_open_l = 1.0
        self.smooth_eye_open_r = 1.0
        self.smooth_eye_x = 0.0
        self.smooth_eye_y = 0.0

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
                    elif name == "tongueOut":
                        face_params["TongueOut"] = score
                    elif name == "cheekPuff":
                        face_params["CheekPuff"] = score
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
