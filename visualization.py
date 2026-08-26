# visualization.py
import cv2
import config

def draw_face_landmarks(frame, landmarks, connections, color=(0, 255, 0), radius=1, line_thickness=1):
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), radius, color, -1)
    for conn in connections:
        idx1, idx2 = conn
        if idx1 < len(points) and idx2 < len(points):
            cv2.line(frame, points[idx1], points[idx2], (0, 255, 255), line_thickness)

def draw_hand_landmarks(frame, landmarks, connections, color=(0, 255, 0), radius=2):
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), radius, color, -1)
    for conn in connections:
        idx1, idx2 = conn
        if idx1 < len(points) and idx2 < len(points):
            cv2.line(frame, points[idx1], points[idx2], (0, 255, 0), 1)

def draw_info(frame, face_params, hand_params, fps, gesture_left=None, gesture_right=None):
    info_text = [
        f"FPS: {fps:.1f}",
        f"Face: {'Found' if face_params.get('faceFound', False) else 'Not found'}",
        f"AngleX: {face_params.get('FaceAngleX', 0):.1f}°",
        f"AngleY: {face_params.get('FaceAngleY', 0):.1f}°",
        f"AngleZ: {face_params.get('FaceAngleZ', 0):.1f}°",
        f"Mouth: {face_params.get('MouthOpen', 0):.2f}",
        f"EyeL: {face_params.get('EyeOpenLeft', 1):.2f} EyeR: {face_params.get('EyeOpenRight', 1):.2f}",
        f"HandL: {'Found' if hand_params.get('HandLeftFound', 0) else '...'} Open: {hand_params.get('HandLeftOpen', 0):.2f}",
        f"HandR: {'Found' if hand_params.get('HandRightFound', 0) else '...'} Open: {hand_params.get('HandRightOpen', 0):.2f}",
    ]
    if gesture_left:
        info_text.append(f"GestureL: {gesture_left}")
    if gesture_right:
        info_text.append(f"GestureR: {gesture_right}")
    y_offset = 30
    for text in info_text:
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20

def show_frame(window_name, frame):
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF == ord('q')
