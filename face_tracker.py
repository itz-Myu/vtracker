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
        self.calib_frames = []

    def _initialize(self):
        base = python.BaseOptions(model_asset_path=self.model_path, delegate=python.BaseOptions.Delegate.GPU)
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

                roll = math.atan2(-r10, r00)   # Z
                pitch = math.atan2(-r21, r22)  # X
                yaw = math.asin(r20)           # Y

                pitch_deg = pitch * 180 / math.pi
                yaw_deg = yaw * 180 / math.pi
                roll_deg = roll * 180 / math.pi
            else:
                pitch_deg = yaw_deg = roll_deg = 0.0

            # --- Калибровка ---
            if not calibration.face_ready:
                self.calib_frames.append((pitch_deg, yaw_deg, roll_deg))
                if len(self.calib_frames) >= config.CALIBRATION_FRAMES:
                    avg_pitch = sum(f[0] for f in self.calib_frames) / len(self.calib_frames)
                    avg_yaw = sum(f[1] for f in self.calib_frames) / len(self.calib_frames)
                    avg_roll = sum(f[2] for f in self.calib_frames) / len(self.calib_frames)
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

            # --- Сглаживание (минимальное) ---
            self.smooth_pitch = utils.smooth_value(self.smooth_pitch, pitch_cal, config.FACE_SMOOTHING)
            self.smooth_yaw = utils.smooth_value(self.smooth_yaw, yaw_cal, config.FACE_SMOOTHING)
            self.smooth_roll = utils.smooth_value(self.smooth_roll, roll_cal, config.FACE_SMOOTHING)

            # --- Маппинг осей ---
            angle_values = {
                "pitch": self.smooth_pitch,
                "yaw": self.smooth_yaw,
                "roll": self.smooth_roll,
            }
            for param, source in config.FACE_AXIS_MAP.items():
                if param == "FaceAngleX" or param == "FaceAngleY":
                    limit = config.FACE_ANGLE_X_LIMIT
                else:
                    limit = config.FACE_ANGLE_Z_LIMIT
                val = angle_values[source]
                if config.FACE_AXIS_INVERT.get(param, False):
                    val = -val
                face_params[param] = utils.clamp(val, -limit, limit)

            # --- Blendshapes ---
            if result.face_blendshapes:
                for b in result.face_blendshapes[0]:
                    name = b.category_name
                    score = b.score
                    if name == "eyeBlinkLeft":
                        face_params["EyeOpenLeft"] = 1.0 - score
                    elif name == "eyeBlinkRight":
                        face_params["EyeOpenRight"] = 1.0 - score
                    elif name == "jawOpen":
                        face_params["MouthOpen"] = score
                    elif name == "mouthSmileLeft":
                        face_params["MouthSmileLeft"] = score
                    elif name == "mouthSmileRight":
                        face_params["MouthSmileRight"] = score
                    elif name == "tongueOut":
                        face_params["TongueOut"] = score
                    elif name == "browInnerUp":
                        face_params["Brows"] = score
                    elif name == "cheekPuff":
                        face_params["CheekPuff"] = score
                    elif name == "noseSneerLeft" or name == "noseSneerRight":
                        face_params["FaceAngry"] = max(face_params.get("FaceAngry", 0), score)
                    elif name == "eyeLookInLeft":
                        face_params["EyeLeftX"] = -score
                    elif name == "eyeLookOutLeft":
                        face_params["EyeLeftX"] = score
                    elif name == "eyeLookUpLeft":
                        face_params["EyeLeftY"] = score
                    elif name == "eyeLookDownLeft":
                        face_params["EyeLeftY"] = -score
                    elif name == "eyeLookInRight":
                        face_params["EyeRightX"] = -score
                    elif name == "eyeLookOutRight":
                        face_params["EyeRightX"] = score
                    elif name == "eyeLookUpRight":
                        face_params["EyeRightY"] = score
                    elif name == "eyeLookDownRight":
                        face_params["EyeRightY"] = -score
                if "MouthSmileLeft" in face_params and "MouthSmileRight" in face_params:
                    face_params["MouthSmile"] = (face_params["MouthSmileLeft"] + face_params["MouthSmileRight"]) / 2
                    del face_params["MouthSmileLeft"]
                    del face_params["MouthSmileRight"]
        else:
            face_params["faceFound"] = False

        return face_params, landmarks

    def close(self):
        if self.landmarker:
            self.landmarker.close()
