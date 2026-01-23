import pymem
import time
import pickle
import math
from pynput.keyboard import Key, Controller

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
GRID_SIZE = 3.0
MIN_Y = 15
LOOKAHEAD_DIST = 6      # how far ahead to aim
STEER_DEADZONE = 0.05
STEER_GAIN = 0.04       # steering sensitivity
DT = 0.05

# -----------------------------
# Load Heatmap
# -----------------------------
with open("track_heatmap.pkl", "rb") as f:
    heatmap = pickle.load(f)

road_cells = set(heatmap["road"].keys())

print(f"✅ Loaded {len(road_cells)} road cells")

# -----------------------------
# Helper Functions
# -----------------------------
def world_to_grid(x, z):
    return int(x / GRID_SIZE), int(z / GRID_SIZE)

def grid_to_world(gx, gz):
    return gx * GRID_SIZE, gz * GRID_SIZE

def nearest_road_cell(x, z):
    gx, gz = world_to_grid(x, z)
    best = None
    best_dist = float("inf")

    for dx in range(-4, 5):
        for dz in range(-4, 5):
            c = (gx + dx, gz + dz)
            if c in road_cells:
                wx, wz = grid_to_world(*c)
                d = math.hypot(wx - x, wz - z)
                if d < best_dist:
                    best = (wx, wz)
                    best_dist = d
    return best

# -----------------------------
# Init
# -----------------------------
pm = pymem.Pymem("TmForever.exe")
keyboard = Controller()

keyboard.press(Key.up)  # gas always on

print("🏎️ Heatmap racing AI started")

try:
    while True:
        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)
        speed = pm.read_int(SPEED_ADDR)

        # Fell off → reset
        if y < MIN_Y:
            keyboard.press(Key.enter)
            time.sleep(0.3)
            keyboard.release(Key.enter)
            time.sleep(2)
            continue

        target = nearest_road_cell(x, z)
        if not target:
            time.sleep(DT)
            continue

        tx, tz = target
        dx = tx - x
        dz = tz - z

        # Steering error
        steer_error = dx * STEER_GAIN

        # Steering control
        if steer_error > STEER_DEADZONE:
            keyboard.press(Key.right)
            keyboard.release(Key.left)
        elif steer_error < -STEER_DEADZONE:
            keyboard.press(Key.left)
            keyboard.release(Key.right)
        else:
            keyboard.release(Key.left)
            keyboard.release(Key.right)

        time.sleep(DT)

except KeyboardInterrupt:
    pass

# Cleanup
keyboard.release(Key.up)
keyboard.release(Key.left)
keyboard.release(Key.right)

print("🛑 Racing AI stopped")
