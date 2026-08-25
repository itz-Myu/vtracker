# utils.py
import math
import numpy as np

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def quaternion_to_euler(x, y, z, w):
    """Преобразование кватерниона в углы Эйлера (radians)"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def smooth_value(current, target, alpha):
    return current * (1 - alpha) + target * alpha

def dead_zone(value, threshold):
    return 0.0 if abs(value) < threshold else value
