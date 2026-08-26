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
    
    last_active_alias = None
    last_gesture_left_alias = None
    last_gesture_right_alias = None
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
            hand_params, gesture_left, gesture_right, hand_landmarks_list = hand_tracker.process(frame_rgb, frame_counter, calib)

            # --- Жесты -> выражения (с отладкой) ---
            def get_alias(gesture, hand):
                if hand == "left":
                    return config.GESTURE_ALIAS_LEFT.get(gesture, config.GESTURE_ALIAS.get(gesture))
                else:
                    return config.GESTURE_ALIAS_RIGHT.get(gesture, config.GESTURE_ALIAS.get(gesture))

            # Функция для безопасной деактивации алиаса
            async def safe_deactivate(alias, reason=""):
                if alias:
                    print(f"🔽 Деактивация '{alias}' ({reason})")
                    await vts.toggle_expression(alias, False)

            # Функция для безопасной активации алиаса
            async def safe_activate(alias, reason=""):
                if alias:
                    print(f"🔼 Активация '{alias}' ({reason})")
                    await vts.toggle_expression(alias, True)

            # --- 1. Проверка двухручного жеста ---
            two_hand_alias = None
            if gesture_left == gesture_right and gesture_left in config.GESTURE_TWO_HAND_ALIAS:
                two_hand_alias = config.GESTURE_TWO_HAND_ALIAS[gesture_left]

            combo_alias = None
            if gesture_left is not None and gesture_right is not None and not two_hand_alias:
                combo = (gesture_left, gesture_right)
                combo_alias = config.GESTURE_COMBINATIONS.get(combo)

            # Определяем, что сейчас активно: двухручный, комбинация или отдельные
            active_alias = two_hand_alias or combo_alias

            if active_alias:
                # Если активный алиас изменился или его не было
                if active_alias != last_active_alias:
                    # Деактивируем всё предыдущее
                    if last_active_alias:
                        await safe_deactivate(last_active_alias, "смена")
                    # Деактивируем отдельные жесты
                    if last_gesture_left:
                        last_alias = get_alias(last_gesture_left, "left")
                        await safe_deactivate(last_alias, "отдельный левый")
                        last_gesture_left = None
                    if last_gesture_right:
                        last_alias = get_alias(last_gesture_right, "right")
                        await safe_deactivate(last_alias, "отдельный правый")
                        last_gesture_right = None
                    # Активируем новый
                    await safe_activate(active_alias, "активация")
                    last_active_alias = active_alias
                # Если активный алиас не изменился, ничего не делаем
            else:
                # Нет активного алиаса (двухручного или комбинации) — обрабатываем отдельные жесты
                if last_active_alias:
                    await safe_deactivate(last_active_alias, "нет активного")
                    last_active_alias = None

                # Левый жест
                if gesture_left is not None:
                    alias = get_alias(gesture_left, "left")
                    if alias and alias != last_gesture_left_alias:
                        if last_gesture_left_alias:
                            await safe_deactivate(last_gesture_left_alias, "смена левого")
                        await safe_activate(alias, "левый")
                        last_gesture_left_alias = alias
                else:
                    if last_gesture_left_alias:
                        await safe_deactivate(last_gesture_left_alias, "левый пропал")
                        last_gesture_left_alias = None

                # Правый жест
                if gesture_right is not None:
                    alias = get_alias(gesture_right, "right")
                    if alias and alias != last_gesture_right_alias:
                        if last_gesture_right_alias:
                            await safe_deactivate(last_gesture_right_alias, "смена правого")
                        await safe_activate(alias, "правый")
                        last_gesture_right_alias = alias
                else:
                    if last_gesture_right_alias:
                        await safe_deactivate(last_gesture_right_alias, "правый пропал")
                        last_gesture_right_alias = None

            # Обновляем старые переменные для совместимости (если нужно)
            last_gesture_left = gesture_left
            last_gesture_right = gesture_right
            
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

                # Формируем строку для отображения жестов
                gesture_display = None
                if gesture_left and gesture_right and gesture_left == gesture_right:
                    gesture_display = f"Both: {gesture_left}"
                elif gesture_left:
                    gesture_display = f"Left: {gesture_left}"
                elif gesture_right:
                    gesture_display = f"Right: {gesture_right}"

                draw_info(frame, face_params, hand_params, fps, gesture_left, gesture_right)
                
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
