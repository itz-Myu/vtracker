# main.py
import asyncio
import argparse
import time
import cv2
from mediapipe.tasks.python.vision import FaceLandmarksConnections

import config
from camera import Camera
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from calibration import Calibration
from vts_client import VTSClient
from visualization import draw_face_landmarks, draw_hand_landmarks, draw_info, show_frame

# Контуры
FACE_CONNECTIONS = [(conn.start, conn.end) for conn in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true', help='Отключить окно визуализации')
    parser.add_argument('--calib-file', default=config.CALIB_FILE, help='Файл калибровки')
    args = parser.parse_args()
    
    calib = Calibration(args.calib_file)
    calib.load()
    
    camera = Camera()
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()
    vts = VTSClient(config.VTS_IP, config.VTS_PORT, config.PLUGIN_NAME, config.PLUGIN_DEVELOPER, config.PLUGIN_TOKEN)
    
    await vts.connect()
    
    last_gesture = None
    frame_counter = 0
    last_time = time.time()
    fps = 0
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break
            
            frame_counter += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- Лицо ---
            face_params, face_landmarks = face_tracker.process(frame_rgb, frame_counter, calib)
            
            # --- Руки ---
            hand_params, gesture, hand_landmarks_list = hand_tracker.process(frame_rgb, frame_counter, calib)
            
            # --- Жесты -> выражения ---
            if gesture and gesture != last_gesture:
                if last_gesture:
                    await vts.toggle_expression(last_gesture, False)
                await vts.toggle_expression(gesture, True)
                last_gesture = gesture
            
            # --- Отправка параметров ---
            param_list = []
            for key, value in face_params.items():
                if key not in ["faceFound"]:
                    param_list.append({"id": key, "value": value})
            for key, value in hand_params.items():
                if key not in ["faceFound"]:
                    param_list.append({"id": key, "value": value})
            await vts.send_parameters(param_list)
            
            # --- Визуализация ---
            if not args.no_gui:
                if face_landmarks is not None:
                    draw_face_landmarks(frame, face_landmarks, FACE_CONNECTIONS)
                if hand_landmarks_list:
                    for lm in hand_landmarks_list:
                        draw_hand_landmarks(frame, lm, HAND_CONNECTIONS)
                draw_info(frame, face_params, hand_params, fps, gesture)
                
                if show_frame(config.WINDOW_NAME, frame):
                    break
            
            # FPS
            if frame_counter % 10 == 0:
                current_time = time.time()
                fps = 10 / (current_time - last_time)
                last_time = current_time
            
            await asyncio.sleep(0.001)
    
    except KeyboardInterrupt:
        print("🛑 Остановка...")
    finally:
        camera.release()
        face_tracker.close()
        hand_tracker.close()
        await vts.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
