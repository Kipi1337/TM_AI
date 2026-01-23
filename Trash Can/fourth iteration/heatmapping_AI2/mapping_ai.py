import pymem
import time
import random
import pickle
import os
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# TrackMania Memory Addresses
# -----------------------------
X_ADDR = 0x1690D470
Y_ADDR = 0x1690D474
Z_ADDR = 0x1690D478
SPEED_ADDR = 0x1665F320

# -----------------------------
# Parameters
# -----------------------------
GRID_SIZE = 3.0          # meters per grid cell
SAFE_SPEED = 80
MIN_Y = 15               # below this = off-track
RUN_TIME = 180           # seconds (3 minutes)

HEATMAP_FILE = "track_heatmap.pkl"
IMAGE_FILE = "track_heatmap.png"

# -----------------------------
# Init
# -----------------------------
pm = pymem.Pymem("TmForever.exe")
keyboard = Controller()

road_map = defaultdict(int)
offtrack_map = defaultdict(int)
wall_map = defaultdict(int)

prev_speed = 0
start_time = time.time()

# Hold gas
keyboard.press(Key.up)

print("🧭 Mapping AI started...")

try:
    while True:
        elapsed = time.time() - start_time
        if elapsed > RUN_TIME:
            break

        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)
        speed = pm.read_int(SPEED_ADDR)

        gx = int(x / GRID_SIZE)
        gz = int(z / GRID_SIZE)
        cell = (gx, gz)

        # Road heatmap
        road_map[cell] += 1

        # Off-track detection
        if y < MIN_Y:
            offtrack_map[cell] += 1
            keyboard.press(Key.enter)
            time.sleep(0.3)
            keyboard.release(Key.enter)
            time.sleep(2)
            continue

        # Wall detection (speed drop)
        if prev_speed - speed > 40 and speed < SAFE_SPEED:
            wall_map[cell] += 1

        prev_speed = speed

        # Steering logic (gentle exploration)
        steer = random.uniform(-0.15, 0.15)

        if speed < SAFE_SPEED:
            steer *= -1

        if steer > 0:
            keyboard.press(Key.right)
            keyboard.release(Key.left)
        else:
            keyboard.press(Key.left)
            keyboard.release(Key.right)

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

# Cleanup
keyboard.release(Key.up)
keyboard.release(Key.left)
keyboard.release(Key.right)

print("🛑 Mapping finished, saving heatmap...")

# Save heatmap data
with open(HEATMAP_FILE, "wb") as f:
    pickle.dump({
        "road": dict(road_map),
        "offtrack": dict(offtrack_map),
        "walls": dict(wall_map)
    }, f)

print(f"✅ Heatmap saved to {HEATMAP_FILE}")
