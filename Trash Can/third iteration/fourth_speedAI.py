import pymem
import time
import torch
from pynput.keyboard import Key, Controller
import random

# -----------------------------
# TrackMania Memory Addresses
# -----------------------------
X_ADDR = 0x1690D470
Y_ADDR = 0x1690D474
Z_ADDR = 0x1690D478
SPEED_ADDR = 0x1665F320
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
    cx = (zone["X"][0] + zone["X"][1]) / 2
    cy = (zone["Y"][0] + zone["Y"][1]) / 2
    cz = (zone["Z"][0] + zone["Z"][1]) / 2
    return ((cx - x)**2 + (cy - y)**2 + (cz - z)**2)**0.5

# -----------------------------
# Reward Function with Speed Bonus
# -----------------------------
def compute_reward(prev_pos, curr_pos, speed, next_cp, checkpoint_reached=False, finished=False):
    # Reward for moving closer to checkpoint
    prev_dist = distance_to_zone(*prev_pos, next_cp)
    curr_dist = distance_to_zone(*curr_pos, next_cp)
    reward = prev_dist - curr_dist  # positive if getting closer

    # Bonus for reaching checkpoint
    if checkpoint_reached:
        reward += 100

    # Bonus for finishing race
    if finished:
        reward += 1000

    # Speed bonus
    if speed > 50:
        reward += 1  # small per-step bonus

    return reward

# -----------------------------
# RL Agent Skeleton
# -----------------------------
actions = ["up", "left", "right", "gas", "none"]
num_actions = len(actions)

# Placeholder random policy
def select_action(state):
    return random.randint(0, num_actions-1)

# -----------------------------
# Attach to TrackMania
# -----------------------------
pm = pymem.Pymem("TmForever.exe")
keyboard = Controller()

# Map AI actions to actual keys
key_map = {
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "gas": Key.up,   # Up arrow is gas in TMNF
    "none": None
}

# Track pressed keys
keys_state = {a: False for a in actions}

# -----------------------------
# RL Environment Loop
# -----------------------------
prev_pos = (0, 0, 0)
next_cp_index = 0

try:
    while True:
        # Read telemetry
        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)
        speed = pm.read_int(SPEED_ADDR)
        turning = pm.read_float(TURN_ADDR)
        racetime = pm.read_int(RACETIME_ADDR)

        curr_pos = (x, y, z)

        # Determine next checkpoint
        checkpoint_reached = False
        next_cp = checkpoints[next_cp_index] if next_cp_index < len(checkpoints) else finish_line
        if in_zone(x, y, z, next_cp):
            checkpoint_reached = True
            next_cp_index += 1

        finished = in_zone(x, y, z, finish_line)

        dist_to_cp = distance_to_zone(x, y, z, next_cp)

        # Build state vector
        state = [x, y, z, speed, turning, dist_to_cp]

        # Compute reward with speed bonus
        reward = compute_reward(prev_pos, curr_pos, speed, next_cp, checkpoint_reached, finished)
        prev_pos = curr_pos

        # Select action (random placeholder)
        action_idx = select_action(state)
        action = actions[action_idx]

        # Inject action safely
        for key_name, pressed in keys_state.items():
            if key_name != action and pressed:
                k = key_map[key_name]
                if k: keyboard.release(k)
                keys_state[key_name] = False

        k = key_map[action]
        if k and not keys_state[action]:
            keyboard.press(k)
            keys_state[action] = True

        # Debug
        print(f"Pos: ({x:.2f},{y:.2f},{z:.2f}) | Speed: {speed} | DistCP: {dist_to_cp:.2f} | Action: {action} | Reward: {reward:.2f}")

        # End episode if finished
        if finished:
            print("Track finished! Reset environment...")
            prev_pos = (0,0,0)
            next_cp_index = 0

        time.sleep(0.05)  # 20 Hz

except KeyboardInterrupt:
    print("RL loop stopped by user")
    # Release all keys
    for key_name, pressed in keys_state.items():
        k = key_map[key_name]
        if pressed and k:
            keyboard.release(k)
