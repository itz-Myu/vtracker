# camera.py
import cv2
import config

class Camera:
    def __init__(self, camera_id=config.CAMERA_ID):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру с индексом {camera_id}")
        self._configure()
        
    def _configure(self):
        fourcc = cv2.VideoWriter_fourcc(*config.CAMERA_FORMAT)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def release(self):
        self.cap.release()
