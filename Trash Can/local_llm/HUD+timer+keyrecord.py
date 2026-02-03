import pygetwindow as gw
import mss
import pytesseract
from PIL import Image
import cv2
import re
import time
import threading
import keyboard

# ================================ #
# Initialization of the HUD reading #
# ================================ #

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
    return gray

def extract_numbers_from_line(text, keyword, expected_count=None):
    for line in text.splitlines():
        if keyword.lower() in line.lower():
            numbers = re.findall(r"-?\d+\.\d+|-?\d+", line)
            values = [float(n) for n in numbers]
            if expected_count is None or len(values) >= expected_count:
                return values
    return None

def parse_trackmania_hud(text):
    data = {}

    # Position
    pos = extract_numbers_from_line(text, "Position", 3)
    if pos:
        data["position"] = tuple(pos[:3])

    # Orientation (Yaw, Pitch, Roll)
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

    # Checkpoints
    cp = extract_numbers_from_line(text, "Checkpoints", 2)
    if cp:
        data["checkpoint_current"] = int(cp[0])
        data["checkpoint_total"] = int(cp[1])

    return data

# ================================ #
# Key Recording                     #
# ================================ #

# Globals
arrow_keys_pressed = []

def on_enter_press(event):
    global enter_count, race_finished, race_started, start_time, race_completed
    if race_finished:
        enter_count += 1
        print("Enter key pressed and captured!")
    elif race_started:
        print("Enter pressed during race — restarting!")
        race_started = False
        race_finished = False
        race_completed = False
        start_time = None
        enter_count = 0
    else:
        print("Enter key pressed, but race has not finished.")

def start_enter_listener():
    """Start Enter key listener in background."""
    keyboard.on_press_key("enter", on_enter_press, suppress=False)

# ================================ #
# Main Loop                         #
# ================================ #

def main():
    global enter_count, race_finished, race_started, start_time, race_completed, arrow_keys_pressed

    enter_count = 0
    window_title = get_window_title()
    crop_coords = (15, 60, 345, 285)  # Adjust as needed

    # Adjustable offset to sync with in-game timer
    TIMER_OFFSET = 0.0
    start_time = None
    race_started = False
    race_finished = False
    race_completed = False

    print("Starting HUD monitoring. Press Ctrl+C to stop...")

    # Start Enter key listener in background
    threading.Thread(target=start_enter_listener, daemon=True).start()

    try:
        while True:
            screenshot = capture_window_screenshot(window_title)
            cropped = crop_image(screenshot, crop_coords)
            processed = preprocess_image(cropped)

            # OCR and parse HUD
            text = pytesseract.image_to_string(processed, lang='eng', config='--psm 6')
            hud = parse_trackmania_hud(text)

            # Poll arrow keys to track multiple keys pressed
            arrow_keys_pressed = []
            for key in ["up", "down", "left", "right"]:
                if keyboard.is_pressed(key):
                    arrow_keys_pressed.append(key)

            # Race reset if speed drops and at first checkpoint
            if race_started and not race_finished:
                if hud.get("speed", 0.0) < 0.01 and hud.get("checkpoint_current", 0) <= 1:
                    print("HUD restart detected — resetting race")
                    race_started = False
                    race_finished = False
                    race_completed = False
                    start_time = None
                    enter_count = 0
                    time.sleep(0.2)
                    continue

            # Detect race start
            if not race_started and hud.get("speed", 0.0) > 0.03:
                start_time = time.time()
                race_started = True
                print("Race started!")

            # Detect race finish
            if race_started and hud.get("checkpoint_current") == hud.get("checkpoint_total"):
                if not race_completed:
                    end_time = time.time()
                    elapsed_time = (end_time - start_time) + TIMER_OFFSET
                    print(f"Race finished! Elapsed time: {elapsed_time:.2f} seconds")
                    race_finished = True
                    race_completed = True

            # Print HUD and all arrow keys currently pressed
            print("=== HUD Data ===")
            for k, v in hud.items():
                print(f"{k}: {v}")
            print("================")
            print(f"Arrow keys pressed: {arrow_keys_pressed}\n")

            # Post-race reset after pressing Enter 3 times
            if race_finished and enter_count >= 3:
                race_finished = False
                race_started = False
                race_completed = False
                start_time = None
                enter_count = 0
                print("Post-race reset")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    main()
