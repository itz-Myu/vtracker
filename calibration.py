# calibration.py
import json
import os

class Calibration:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {
            "face": {"FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0},
            "hand_left": {"AngleX": 0.0, "AngleZ": 0.0},
            "hand_right": {"AngleX": 0.0, "AngleZ": 0.0},
        }
        self.face_ready = False
        self.hand_ready = False
        
    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
                self.face_ready = True
                self.hand_ready = True
                print("📂 Калибровка загружена из файла")
                return True
        return False
    
    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        print("💾 Калибровка сохранена")
    
    def get_face_offset(self):
        return self.data["face"]
    
    def set_face_offset(self, pitch, yaw, roll):
        self.data["face"]["FaceAngleX"] = pitch
        self.data["face"]["FaceAngleY"] = yaw
        self.data["face"]["FaceAngleZ"] = roll
        self.face_ready = True
        self.save()
    
    def get_hand_offset(self, side="left"):
        return self.data[f"hand_{side}"]
    
    def set_hand_offset(self, side, angle_x, angle_z):
        self.data[f"hand_{side}"]["AngleX"] = angle_x
        self.data[f"hand_{side}"]["AngleZ"] = angle_z
        self.hand_ready = True
        self.save()
