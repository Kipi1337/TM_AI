import pygetwindow as gw
import mss
import pytesseract
from PIL import Image
import cv2
import re
import time
import numpy as np

# =========================
# Helper functions
# =========================

def get_window_title():
    return "Trackmania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.10)"

def capture_window_screenshot(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise ValueError(f"Window with title '{window_title}' not found.")
    window = windows[0]

    left, top, width, height = window.left, window.top, window.width, window.height

    with mss.mss() as sct:
        screenshot = sct.grab((left, top, left + width, top + height))
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    return img

def crop_image(image, crop_coords):
    return image.crop(crop_coords)

def preprocess_image(image):
    gray = image.convert('L')  # Grayscale
    # Optional: simple threshold to remove noise
    np_img = cv2.cvtColor(np.array(gray), cv2.COLOR_GRAY2BGR)
    return gray

def extract_numbers_from_line(text, keyword, expected_count=None):
    """
    Finds a line containing `keyword` and extracts float numbers from it.
    Returns a list of floats or None.
    """
    for line in text.splitlines():
        if keyword.lower() in line.lower():
            numbers = re.findall(r"-?\d+\.\d+|-?\d+", line)
            values = [float(n) for n in numbers]
            if expected_count is None or len(values) >= expected_count:
                return values
    return None

def parse_trackmania_hud(text):
    """
    Parses OCR text from Trackmania overlay into variables.
    """
    data = {}

    # Position
    pos = extract_numbers_from_line(text, "Position", 3)
    if pos:
        data["position"] = tuple(pos[:3])

    # Velocity
    vel = extract_numbers_from_line(text, "Velocity", 3)
    if vel:
        data["velocity"] = tuple(vel[:3])

    # Orientation (Yaw, Pitch, Roll on same line)
    for line in text.splitlines():
        if "Yaw" in line and "Pitch" in line and "Roll" in line:
            numbers = re.findall(r"-?\d+\.\d+|-?\d+", line)
            if len(numbers) >= 3:
                data["yaw"] = float(numbers[0])
                data["pitch"] = float(numbers[1])
                data["roll"] = float(numbers[2])
            break

    # Speed
    speed = extract_numbers_from_line(text, "Real Speed", 1)
    data["speed"] = speed[0] if speed else 0.0

    # Steering
    steer = extract_numbers_from_line(text, "Steer Direction", 1)
    data["steer_direction"] = steer[0] if steer else None

    turn_rate = extract_numbers_from_line(text, "Turning Rate", 1)
    data["turning_rate"] = turn_rate[0] if turn_rate else None

    # Checkpoints (0/3)
    cp = extract_numbers_from_line(text, "Checkpoints", 2)
    if cp:
        data["checkpoint_current"] = int(cp[0])
        data["checkpoint_total"] = int(cp[1])

    return data


# =========================
# Main loop
# =========================

def main():
    window_title = get_window_title()
    
    # Crop coordinates for HUD overlay (adjust as needed)
    crop_coords = (15, 60, 345, 285)  # left, top, right, bottom

    print("Starting HUD monitoring. Press Ctrl+C to stop...")
    try:
        while True:
            screenshot = capture_window_screenshot(window_title)
            cropped = crop_image(screenshot, crop_coords)
            processed = preprocess_image(cropped)

            # Perform OCR
            text = pytesseract.image_to_string(processed, lang='eng', config='--psm 6')
            
            # Parse numeric values
            hud = parse_trackmania_hud(text)

            # Print all variables
            print("=== HUD Data ===")
            for k, v in hud.items():
                print(f"{k}: {v}")
            print("================\n")

            time.sleep(1)  # 1 frame per second (adjust as needed)
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    main()
