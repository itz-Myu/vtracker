# gesture_recognition.py
import math
import config

def get_thresholds(finger_index):
    """
    Возвращает пороги (extended, folded) для пальца по его индексу.
    Индексы: 0 - большой, 1 - указательный, 2 - средний, 3 - безымянный, 4 - мизинец.
    """
    if finger_index == 0:
        return config.GESTURE_THUMB_EXTENDED, config.GESTURE_THUMB_FOLDED
    else:
        return config.GESTURE_THRESHOLD_EXTENDED, config.GESTURE_THRESHOLD_FOLDED

def is_extended(finger_open, index):
    """Проверяет, разогнут ли палец с учётом его индивидуального порога."""
    ext, _ = get_thresholds(index)
    return finger_open[index] > ext

def is_folded(finger_open, index):
    """Проверяет, согнут ли палец с учётом его индивидуального порога."""
    _, fold = get_thresholds(index)
    return finger_open[index] < fold

def recognize_gesture(finger_open_left, landmarks_left, finger_open_right=None, landmarks_right=None):
    """
    Распознаёт жест по значениям открытия пальцев и координатам.
    finger_open = [thumb, index, middle, ring, pinky] (0..1)
    Возвращает строку с именем жеста или None.
    """
    # --- Вспомогательные функции с учётом индивидуальных порогов ---
    def extended_indices(finger_open):
        return [i for i in range(5) if is_extended(finger_open, i)]

    def folded_indices(finger_open):
        return [i for i in range(5) if is_folded(finger_open, i)]

    def all_folded(finger_open, indices):
        return all(is_folded(finger_open, i) for i in indices)

    def all_extended(finger_open, indices):
        return all(is_extended(finger_open, i) for i in indices)

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
        if set(extended) == {1, 2} and all_folded(finger_open_left, [0, 3, 4]):
            return "peace"
        # OK (👌) — указательный, средний, безымянный и мизинец разогнуты.
        thumb_tip = landmarks_left[4]
        index_tip = landmarks_left[8]
        wrist = landmarks_left[0]
        middle_mcp = landmarks_left[9]

        dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )
        palm_size = math.sqrt(
            (wrist.x - middle_mcp.x)**2 + 
            (wrist.y - middle_mcp.y)**2 + 
            (wrist.z - middle_mcp.z)**2
        )

        if palm_size > 1e-6 and (dist / palm_size) < 0.45:
            # Пальцы касаются. Теперь проверяем, что остальные не сжаты в кулак.
            # Достаточно, чтобы они были чуть разогнуты (больше 0.2), 
            # так как при нормальном ОК указательный сгибается, а мизинец часто скручен.
            if (finger_open_left[2] > 0.2 and 
                finger_open_left[3] > 0.2 and 
                finger_open_left[4] > 0.2):
                return "ok"
        # Большой палец вверх (👍)
        if set(extended) == {0} and all_folded(finger_open_left, [1, 2, 3, 4]):
            return "thumbs_up"
        # Средний палец (🖕)
        if set(extended) == {2} and all_folded(finger_open_left, [0, 1, 3, 4]):
            return "middle_finger"
        # Указательный вверх (☝️)
        if set(extended) == {1} and all_folded(finger_open_left, [0, 2, 3, 4]):
            return "point_up"
        # Мизинец вверх (🤙)
        if set(extended) == {4} and all_folded(finger_open_left, [0, 1, 2, 3]):
            return "pinky_up"
        # Рога (🤘) — указательный + мизинец
        if set(extended) == {1, 4} and all_folded(finger_open_left, [0, 2, 3]):
            return "horns"
        # Три пальца (указательный, средний, безымянный)
        if set(extended) == {1, 2, 3} and all_folded(finger_open_left, [0, 4]):
            return "three_fingers"
        # Четыре пальца (без большого)
        if set(extended) == {1, 2, 3, 4} and all_folded(finger_open_left, [0]):
            return "four_fingers"
        return None

    # --- ДВЕ РУКИ ---
    else:
        left_index = landmarks_left[8]
        right_index = landmarks_right[8]
        dist_index = math.sqrt((left_index.x - right_index.x)**2 + (left_index.y - right_index.y)**2)

        # 👉👈 (Fingers together) — указательные близко, остальные согнуты
        if dist_index < config.GESTURE_DISTANCE_THRESHOLD:
            if (is_folded(finger_open_left, 0) and is_folded(finger_open_left, 2) and
                is_folded(finger_open_left, 3) and is_folded(finger_open_left, 4) and
                is_folded(finger_open_right, 0) and is_folded(finger_open_right, 2) and
                is_folded(finger_open_right, 3) and is_folded(finger_open_right, 4)):
                return "fingers_together"

        # 🫶 (Heart) — большие и указательные близко и разогнуты (стандартный порог 0.65)
        left_thumb = landmarks_left[4]
        right_thumb = landmarks_right[4]
        dist_thumb = math.sqrt((left_thumb.x - right_thumb.x)**2 + (left_thumb.y - right_thumb.y)**2)
        if (dist_thumb < config.GESTURE_DISTANCE_THRESHOLD and
            dist_index < config.GESTURE_DISTANCE_THRESHOLD and
            is_extended(finger_open_left, 0) and is_extended(finger_open_left, 1) and
            is_extended(finger_open_right, 0) and is_extended(finger_open_right, 1)):
            return "heart"

        return None
