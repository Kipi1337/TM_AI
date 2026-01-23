import mss
import pytesseract
from PIL import Image
import re
import time
import pygetwindow as gw

# -----------------------------
# Find TrackMania window
# -----------------------------
window_title = "TrackMania Nations Forever (TMInterface 1.4.3)"
try:
    tm_window = gw.getWindowsWithTitle(window_title)[0]
except IndexError:
    raise RuntimeError("TrackMania window not found!")

# Coordinates of telemetry overlay relative to window top-left
# You may need to adjust these for your resolution
overlay_offset = {"top": 50, "left": 50, "width": 300, "height": 150}

# -----------------------------
# OCR Helper Functions
# -----------------------------
def parse_telemetry(text):
    """Extract telemetry values from OCR text."""
    data = {}

    pos_match = re.search(r"Position:\s*([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)", text)
    if pos_match:
        data['x'], data['y'], data['z'] = map(float, pos_match.groups())

    vel_match = re.search(r"Velocity:\s*([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)", text)
    if vel_match:
        data['vx'], data['vy'], data['vz'] = map(float, vel_match.groups())

    speed_match = re.search(r"Real Speed:\s*([\d.-]+)", text)
    if speed_match:
        data['speed'] = float(speed_match.group(1))

    racetime_match = re.search(r"Race Time:\s*([\d]+)", text)
    if racetime_match:
        data['racetime'] = int(racetime_match.group(1))

    return data

# -----------------------------
# Real-time Capture Loop
# -----------------------------
with mss.mss() as sct:
    while True:
        # Compute absolute monitor coordinates based on window
        top = tm_window.top + overlay_offset["top"]
        left = tm_window.left + overlay_offset["left"]
        width = overlay_offset["width"]
        height = overlay_offset["height"]

        monitor = {"top": top, "left": left, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

        # OCR
        text = pytesseract.image_to_string(img)
        telemetry = parse_telemetry(text)

        if telemetry:
            print(f"Telemetry: {telemetry}")
        else:
            print("No telemetry detected.")

        time.sleep(0.1)
