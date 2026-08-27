# camera.py
import cv2
import threading
import time
import config

class Camera:
    def __init__(self, camera_id=config.CAMERA_ID):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру с индексом {camera_id}")
        self._configure()
        self.last_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        while self.last_frame is None:
            time.sleep(0.01)

    def _configure(self):
        fourcc = cv2.VideoWriter_fourcc(*config.CAMERA_FORMAT)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.last_frame = frame
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return self.last_frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
