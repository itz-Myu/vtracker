# gesture_recognition.py
import math

def recognize_gesture(finger_open_left, landmarks_left, finger_open_right=None, landmarks_right=None):
    """
    Распознаёт жест по значениям открытия пальцев и координатам.
    finger_open = [thumb, index, middle, ring, pinky] (0..1)
    Возвращает строку с именем жеста или None.
    """
    # Вспомогательные функции
    def extended_indices(finger_open):
        return [i for i, v in enumerate(finger_open) if v > 0.7]
    
    def folded_indices(finger_open):
        return [i for i, v in enumerate(finger_open) if v < 0.3]
    
    # --- ОДНА РУКА ---
    if finger_open_right is None or landmarks_right is None:
        extended = extended_indices(finger_open_left)
        folded = folded_indices(finger_open_left)
        
        # Кулак
        if len(folded) == 5:
            return "fist"
        # Открытая ладонь
        if len(extended) == 5:
            return "open_palm"
        # Мир (✌️)
        if set(extended) == {1, 2} and all(v < 0.3 for v in [finger_open_left[0], finger_open_left[3], finger_open_left[4]]):
            return "peace"
        # OK (👌) — проверяем расстояние между большим и указательным
        if set(extended) == {0, 1} and all(v > 0.7 for v in [finger_open_left[2], finger_open_left[3], finger_open_left[4]]):
            thumb_tip = landmarks_left[4]
            index_tip = landmarks_left[8]
            dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            if dist < 0.05:
                return "ok"
        # Большой палец вверх (👍)
        if set(extended) == {0} and all(v < 0.3 for v in finger_open_left[1:]):
            return "thumbs_up"
        # Средний палец (🖕)
        if set(extended) == {2} and all(v < 0.3 for v in [finger_open_left[0], finger_open_left[1], finger_open_left[3], finger_open_left[4]]):
            return "middle_finger"
        # Указательный вверх (☝️)
        if set(extended) == {1} and all(v < 0.3 for v in [finger_open_left[0], finger_open_left[2], finger_open_left[3], finger_open_left[4]]):
            return "point_up"
        # Мизинец вверх (🤙)
        if set(extended) == {4} and all(v < 0.3 for v in [finger_open_left[0], finger_open_left[1], finger_open_left[2], finger_open_left[3]]):
            return "pinky_up"
        # Рога (🤘) — указательный + мизинец
        if set(extended) == {1, 4} and all(v < 0.3 for v in [finger_open_left[0], finger_open_left[2], finger_open_left[3]]):
            return "horns"
        # Три пальца (указательный, средний, безымянный)
        if set(extended) == {1, 2, 3} and all(v < 0.3 for v in [finger_open_left[0], finger_open_left[4]]):
            return "three_fingers"
        # Четыре пальца (без большого)
        if set(extended) == {1, 2, 3, 4} and all(v < 0.3 for v in [finger_open_left[0]]):
            return "four_fingers"
        return None
    
    # --- ДВЕ РУКИ ---
    else:
        # 👉👈 (Fingers together) — указательные пальцы близко
        left_index = landmarks_left[8]
        right_index = landmarks_right[8]
        dist_index = math.sqrt((left_index.x - right_index.x)**2 + (left_index.y - right_index.y)**2)
        if dist_index < 0.05 and finger_open_left[1] > 0.7 and finger_open_right[1] > 0.7:
            return "fingers_together"
        
        # 🫶 (Heart / Triangle) — большие и указательные образуют треугольник
        left_thumb = landmarks_left[4]
        right_thumb = landmarks_right[4]
        left_index = landmarks_left[8]
        right_index = landmarks_right[8]
        dist_thumb = math.sqrt((left_thumb.x - right_thumb.x)**2 + (left_thumb.y - right_thumb.y)**2)
        dist_index = math.sqrt((left_index.x - right_index.x)**2 + (left_index.y - right_index.y)**2)
        if dist_thumb < 0.05 and dist_index < 0.05 and finger_open_left[0] > 0.7 and finger_open_left[1] > 0.7 and finger_open_right[0] > 0.7 and finger_open_right[1] > 0.7:
            return "heart"
        
        return None
