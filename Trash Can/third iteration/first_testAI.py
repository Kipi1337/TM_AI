import pymem
import time
import csv

# -----------------------------
# TrackMania Memory Addresses
# -----------------------------
X_ADDR = 0x1690D470
Y_ADDR = 0x1690D474
Z_ADDR = 0x1690D478
SPEED_ADDR = 0x1665F320       # int pointer; adjust if needed
TURN_ADDR = 0x1690D4BC
RACETIME_ADDR = 0x1665F290

# -----------------------------
# Checkpoints / Finish Line
# -----------------------------
checkpoints = [
    {"X": (967, 977), "Y": (19, 29), "Z": (300, 310)},   # 1st checkpoint
    {"X": (492, 502), "Y": (78, 88), "Z": (171, 181)},   # 2nd checkpoint
]
finish_line = {"X": (17, 27), "Y": (108, 118), "Z": (172, 182)}

def in_zone(x, y, z, zone):
    return (zone["X"][0] <= x <= zone["X"][1] and
            zone["Y"][0] <= y <= zone["Y"][1] and
            zone["Z"][0] <= z <= zone["Z"][1])

def distance_to_zone(x, y, z, zone):
    # Euclidean distance to center of the zone
    cx = (zone["X"][0] + zone["X"][1]) / 2
    cy = (zone["Y"][0] + zone["Y"][1]) / 2
    cz = (zone["Z"][0] + zone["Z"][1]) / 2
    return ((cx - x)**2 + (cy - y)**2 + (cz - z)**2)**0.5

# -----------------------------
# Load Human Input File
# -----------------------------
human_input_file = "human_run.txt"
human_actions = []

with open(human_input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line: 
            continue
        # Example: 0.46-29.94 press up
        times, action = line.split(' press ')
        start, end = map(float, times.split('-'))
        human_actions.append({"start_ms": int(start*1000),
                              "end_ms": int(end*1000),
                              "action": action})

def get_human_action_at_time(ms):
    for act in human_actions:
        if act["start_ms"] <= ms <= act["end_ms"]:
            return act["action"]
    return "none"  # no input

# -----------------------------
# Output CSV File
# -----------------------------
output_file = "tm_ai_training_ready.csv"

# -----------------------------
# Attach to TrackMania
# -----------------------------
pm = pymem.Pymem("TmForever.exe")

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(["racetime_ms", "x", "y", "z", "speed", "turning", "dist_to_next_cp", "action"])

    try:
        while True:
            # Read telemetry
            racetime = pm.read_int(RACETIME_ADDR)
            x = pm.read_float(X_ADDR)
            y = pm.read_float(Y_ADDR)
            z = pm.read_float(Z_ADDR)
            speed = pm.read_int(SPEED_ADDR)
            turning = pm.read_float(TURN_ADDR)

            # Determine next checkpoint
            next_cp = None
            for cp in checkpoints:
                if not in_zone(x, y, z, cp):
                    next_cp = cp
                    break
            if next_cp is None:
                next_cp = finish_line

            dist_to_next_cp = distance_to_zone(x, y, z, next_cp)

            # Get human action
            action = get_human_action_at_time(racetime)

            # Save row
            writer.writerow([racetime, x, y, z, speed, turning, dist_to_next_cp, action])

            # Debug print
            print(f"Time: {racetime} ms | Pos: ({x:.2f},{y:.2f},{z:.2f}) | Speed: {speed} | Turn: {turning:.2f} | DistCP: {dist_to_next_cp:.2f} | Action: {action}")

            # Sample at 20 Hz
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Recording stopped by user")
