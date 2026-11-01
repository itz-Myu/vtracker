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
    parser.add_argument('--no-gui',     action='store_true',       help='Отключить окно визуализации')
    parser.add_argument('--no-hands',   action='store_true',       help='Отключить трекинг рук')
    parser.add_argument('--calib-file', default=config.CALIB_FILE, help='Файл калибровки')
    args = parser.parse_args()

    calib = Calibration(args.calib_file)
    calib.load()

    camera = Camera()
    face_tracker = FaceTracker()
    hand_tracker = None if args.no_hands else HandTracker()
    vts = VTSClient(config.VTS_IP, config.VTS_PORT, config.PLUGIN_NAME, config.PLUGIN_DEVELOPER, config.PLUGIN_TOKEN)

    await vts.connect()

    last_active_alias = None
    last_gesture_left_alias = None
    last_gesture_right_alias = None

    # История жестов для стабильности (защита от фантомных рук и моргания жестов)
    gesture_history_left = []
    gesture_history_right = []

    # Троттлинг для VTS
    last_param_send_time = 0
    frame_counter = 0
    last_time = time.time()
    fps = 0
    
    try:
        target_fps = 60
        target_frame_time = 1 / target_fps

        while True:
            frame_start_time = time.time()
            frame = camera.get_frame()
            if frame is None:
                break

            frame_counter += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Проверка подключения к VTS ---
            if not vts.connected:
                print("🔄 VTS отключён, попытка переподключения...")
                try:
                    await vts.connect()
                except Exception as e:
                    print(f"❌ Не удалось переподключиться: {e}")
                    await asyncio.sleep(1)
                    continue

            # --- Лицо ---
            face_params, face_landmarks = face_tracker.process(frame_rgb, frame_counter, calib)

            # --- Руки ---
            if not args.no_hands:
                hand_params, gesture_left, gesture_right, hand_landmarks_list = hand_tracker.process(frame_rgb, frame_counter, calib)
            else:
                hand_params, gesture_left, gesture_right, hand_landmarks_list = {}, None, None, []

            # --- Жесты -> выражения (с историей для стабильности) ---

            # Добавляем текущие жесты в историю (храним последние 5 кадров)
            gesture_history_left.append(gesture_left)
            gesture_history_right.append(gesture_right)
            if len(gesture_history_left) > 5: gesture_history_left.pop(0)
            if len(gesture_history_right) > 5: gesture_history_right.pop(0)

            # Жест считается стабильным, если он встретился хотя бы 3 раза из 5
            def get_stable_gesture(history):
                for g in history:
                    if g is not None and history.count(g) >= 3:
                        return g
                return None

            stable_left = get_stable_gesture(gesture_history_left)
            stable_right = get_stable_gesture(gesture_history_right)

            # --- Определяем, что сейчас активно ---
            def get_alias(gesture, hand):
                if hand == "left":
                    return config.GESTURE_ALIAS_LEFT.get(gesture, config.GESTURE_ALIAS.get(gesture))
                else:
                    return config.GESTURE_ALIAS_RIGHT.get(gesture, config.GESTURE_ALIAS.get(gesture))

            async def safe_toggle(alias, active, reason=""):
                if alias:
                    if active:
                        print(f"🔼 Активация '{alias}' ({reason})")
                    else:
                        print(f"🔽 Деактивация '{alias}' ({reason})")
                    await vts.toggle_expression(alias, active)

            # --- Проверка двухручного жеста (только если оба стабильны) ---
            two_hand_alias = None
            if stable_left is not None and stable_right is not None and stable_left == stable_right:
                two_hand_alias = config.GESTURE_TWO_HAND_ALIAS.get(stable_left)

            # --- Проверка комбинации (если оба стабильны и не двухручные) ---
            combo_alias = None
            if stable_left is not None and stable_right is not None and not two_hand_alias:
                combo = (stable_left, stable_right)
                combo_alias = config.GESTURE_COMBINATIONS.get(combo)

            # --- Определяем активный алиас (приоритет: двухручный > комбинация) ---
            new_active_alias = two_hand_alias or combo_alias

            # --- Обработка смены активного алиаса ---
            if new_active_alias:
                if new_active_alias != last_active_alias:
                    # Деактивируем предыдущий активный
                    if last_active_alias:
                        await safe_toggle(last_active_alias, False, "смена активного")
                    # Деактивируем отдельные жесты
                    if last_gesture_left_alias:
                        await safe_toggle(last_gesture_left_alias, False, "отдельный левый (активный)")
                        last_gesture_left_alias = None
                    if last_gesture_right_alias:
                        await safe_toggle(last_gesture_right_alias, False, "отдельный правый (активный)")
                        last_gesture_right_alias = None
                    # Активируем новый
                    await safe_toggle(new_active_alias, True, "активация")
                    last_active_alias = new_active_alias
                # Если активный не изменился, ничего не делаем
            else:
                # Нет активного — обрабатываем отдельные жесты
                if last_active_alias:
                    await safe_toggle(last_active_alias, False, "нет активного")
                    last_active_alias = None

                # --- Левый жест ---
                if stable_left is not None:
                    alias = get_alias(stable_left, "left")
                    if alias and alias != last_gesture_left_alias:
                        if last_gesture_left_alias:
                            await safe_toggle(last_gesture_left_alias, False, "смена левого")
                        await safe_toggle(alias, True, "левый")
                        last_gesture_left_alias = alias
                else:
                    if last_gesture_left_alias:
                        await safe_toggle(last_gesture_left_alias, False, "левый пропал")
                        last_gesture_left_alias = None

                # --- Правый жест ---
                if stable_right is not None:
                    alias = get_alias(stable_right, "right")
                    if alias and alias != last_gesture_right_alias:
                        if last_gesture_right_alias:
                            await safe_toggle(last_gesture_right_alias, False, "смена правого")
                        await safe_toggle(alias, True, "правый")
                        last_gesture_right_alias = alias
                else:
                    if last_gesture_right_alias:
                        await safe_toggle(last_gesture_right_alias, False, "правый пропал")
                        last_gesture_right_alias = None

            # --- Отправка параметров ---
            param_list = []
            for key, value in face_params.items():
                if key not in ["faceFound"]:
                    param_list.append({"id": key, "value": value})
            for key, value in hand_params.items():
                if key not in ["faceFound"]:
                    param_list.append({"id": key, "value": value})
            # Троттлинг
            current_time = time.time()
            if current_time - last_param_send_time >= 0.033:
                await vts.send_parameters(param_list)
                last_param_send_time = current_time

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

            # FPS ограничение
            elapsed = time.time() - frame_start_time
            if elapsed < target_frame_time:
                await asyncio.sleep(target_frame_time - elapsed)
            else:
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
