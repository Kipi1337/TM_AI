import pymem
import time
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Controller
from collections import defaultdict

# =============================
# TrackMania Memory Addresses
# =============================
X_ADDR = 0x1690D470
Y_ADDR = 0x1690D474
Z_ADDR = 0x1690D478
SPEED_ADDR = 0x1665F320

# =============================
# Config
# =============================
GRID_SIZE = 3.0          # meters per cell
DT = 0.05
SAFE_SPEED = 40
MAX_RUNTIME = 300        # seconds (5 minutes)

HEATMAP_FILE = "heatmap.pkl"
IMAGE_FILE = "heatmap.png"

# =============================
# Init
# =============================
pm = pymem.Pymem("TmForever.exe")
keyboard = Controller()

road_heatmap = defaultdict(int)
offtrack_heatmap = defaultdict(int)

start_time = time.time()

# =============================
# Helpers
# =============================
def grid_cell(x, z):
    return (int(x / GRID_SIZE), int(z / GRID_SIZE))

# =============================
# Start driving
# =============================
print("Mapping AI started...")

keyboard.press(Key.up)  # Always gas

try:
    while True:
        # Stop after max runtime
        if time.time() - start_time > MAX_RUNTIME:
            print("Mapping complete.")
            break

        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)
        speed = pm.read_int(SPEED_ADDR)

        cell = grid_cell(x, z)

        # Record road / off-track
        if y > 15:
            road_heatmap[cell] += 1
        else:
            offtrack_heatmap[cell] += 1

        # Gentle exploration steering
        steer_strength = 0.15

        if speed < SAFE_SPEED:
            steer = random.choice([-1, 1]) * steer_strength
        else:
            steer = random.uniform(-steer_strength, steer_strength)

        if steer < 0:
            keyboard.press(Key.left)
            keyboard.release(Key.right)
        else:
            keyboard.press(Key.right)
            keyboard.release(Key.left)

        time.sleep(DT)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    keyboard.release(Key.up)
    keyboard.release(Key.left)
    keyboard.release(Key.right)

# =============================
# Save Heatmap Data
# =============================
with open(HEATMAP_FILE, "wb") as f:
    pickle.dump({
        "road": dict(road_heatmap),
        "offtrack": dict(offtrack_heatmap),
        "grid_size": GRID_SIZE
    }, f)

print("Heatmap data saved.")

# =============================
# Visualization
# =============================
if road_heatmap:
    xs = [c[0] for c in road_heatmap]
    zs = [c[1] for c in road_heatmap]
    values = [road_heatmap[c] for c in road_heatmap]

    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)

    grid = np.zeros((x_max - x_min + 1, z_max - z_min + 1))

    for (x, z), v in road_heatmap.items():
        grid[x - x_min, z - z_min] = v

    plt.figure(figsize=(10, 8))
    plt.imshow(grid.T, origin="lower", cmap="hot")
    plt.colorbar(label="Visits")
    plt.title("TrackMania A01 – Road Heatmap")
    plt.xlabel("X")
    plt.ylabel("Z")

    plt.savefig(IMAGE_FILE, dpi=200)
    plt.show()

    print(f"Heatmap image saved as {IMAGE_FILE}")
