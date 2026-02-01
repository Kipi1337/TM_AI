import os
import time
import random
import pickle
import numpy as np
from collections import deque
from pynput.keyboard import Key, Controller
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pygetwindow as gw
import mss
from PIL import Image
import pytesseract
import cv2
import re

# =========================
# CONFIG
# =========================
STATE_SIZE = 11
ACTION_SIZE = 7

REPLAY_FILE = "replay_buffer.pkl"
MODEL_FILE = "dqn_model.keras"

TRAIN_MODE = True
MAX_MEMORY = 100_000
BATCH_SIZE = 64
TARGET_UPDATE_EVERY = 5
ACTION_DELAY = 0.15
STUCK_SPEED_THRESHOLD = 5.0
STUCK_TIME_LIMIT = 999999
FALL_Z_THRESHOLD = 15.0
FALL_RESET_ENABLED = True

WINDOW_TITLE = "Trackmania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.10)"
HUD_CROP = (15, 60, 345, 285)

FORCED_CHECKPOINT_TOTAL = 3  # Force the total checkpoints for this track

keyboard_controller = Controller()

# =========================
# ACTIONS
# =========================
def press_action(action):
    for k in [Key.up, Key.down, Key.left, Key.right]:
        try:
            keyboard_controller.release(k)
        except:
            pass

    if action == 0:
        keyboard_controller.press(Key.up)
    elif action == 1:
        keyboard_controller.press(Key.down)
    elif action == 2:
        keyboard_controller.press(Key.left)
    elif action == 3:
        keyboard_controller.press(Key.right)
    elif action == 4:
        keyboard_controller.press(Key.up)
        keyboard_controller.press(Key.left)
    elif action == 5:
        keyboard_controller.press(Key.up)
        keyboard_controller.press(Key.right)
    # 6 = no input

# =========================
# HUD → STATE
# =========================
def hud_to_state(hud):
    pos = hud.get("position", (0, 0, 0))
    state = np.array([
        pos[0], pos[1], pos[2],
        hud.get("speed", 0.0),
        hud.get("yaw", 0.0),
        hud.get("pitch", 0.0),
        hud.get("roll", 0.0),
        hud.get("steer_direction", 0.0),
        hud.get("turning_rate", 0.0),
        hud.get("checkpoint_current", 0),
        hud.get("checkpoint_total", 1)
    ], dtype=np.float32)
    return np.clip(state, -1000, 1000).reshape(1, -1)

# =========================
# REWARD & DONE
# =========================
def compute_reward(prev_hud, hud, finished):
    reward = 0.0
    speed = hud.get("speed", 0.0)
    prev_speed = prev_hud.get("speed", 0.0)

    reward += speed * 0.05
    reward += (speed - prev_speed) * 0.1

    if hud.get("checkpoint_current", 0) > prev_hud.get("checkpoint_current", 0):
        reward += 200

    if speed < 1.0:
        reward -= 2.0

    if prev_speed > 5 and speed < 0.5:
        reward -= 50.0

    if finished:
        reward += 1000.0

    return reward

def is_done(hud):
    # Placeholder; we use a buffer-based check in main loop
    return False

# =========================
# HUD OCR
# =========================
def get_window():
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        raise RuntimeError(f"Window '{WINDOW_TITLE}' not found")
    return windows[0]

def capture_screenshot(window):
    left, top, width, height = window.left, window.top, window.width, window.height
    with mss.mss() as sct:
        img = sct.grab((left, top, left + width, top + height))
    return Image.frombytes("RGB", img.size, img.rgb)

def preprocess_for_ocr(img):
    gray = img.convert("L")
    np_img = np.array(gray)
    _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def extract_numbers(text):
    nums = re.findall(r"-?\d+\.\d+|-\d+|\d+", text)
    try:
        return [float(n) for n in nums]
    except ValueError:
        return []

def read_hud(window):
    screenshot = capture_screenshot(window)
    hud_img = screenshot.crop(HUD_CROP)
    processed = preprocess_for_ocr(hud_img)
    text = pytesseract.image_to_string(processed, lang="eng", config="--psm 6")
    nums = extract_numbers(text)
    
    data = {
        "position": tuple(nums[0:3]) if len(nums) >= 3 else (0,0,0),
        "velocity": tuple(nums[3:6]) if len(nums) >= 6 else (0,0,0),
        "yaw": nums[6] if len(nums) >= 7 else 0.0,
        "pitch": nums[7] if len(nums) >= 8 else 0.0,
        "roll": nums[8] if len(nums) >= 9 else 0.0,
        "speed": nums[12] if len(nums) >= 13 else 0.0,
        "steer_direction": nums[14] if len(nums) >= 15 else 0.0,
        "turning_rate": nums[15] if len(nums) >= 16 else 0.0,
        "checkpoint_current": int(nums[16]) if len(nums) >= 17 else 0,
        "checkpoint_total": FORCED_CHECKPOINT_TOTAL
    }

    # ---- DEBUG PRINT ----
#    print("\n--- HUD DEBUG ---")
#    for key, value in data.items():
#        print(f"{key}: {value}")
#    print("--- END HUD DEBUG ---\n")

    return data

# =========================
# DQN AGENT
# =========================
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEMORY)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 0.001

        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        if os.path.exists(MODEL_FILE):
            self.model.load_weights(MODEL_FILE)
            self.target_model.set_weights(self.model.get_weights())
            print("Loaded model from disk")

        if os.path.exists(REPLAY_FILE):
            with open(REPLAY_FILE, "rb") as f:
                self.memory = pickle.load(f)
            print(f"Loaded replay buffer ({len(self.memory)})")

    def _build_model(self):
        model = Sequential([
            Dense(128, activation="relu", input_shape=(STATE_SIZE,)),
            Dense(128, activation="relu"),
            Dense(ACTION_SIZE, activation="linear")
        ])
        model.compile(optimizer=Adam(self.lr), loss="mse")
        return model

    def act(self, state):
        if TRAIN_MODE and random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def remember(self, exp):
        self.memory.append(exp)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if TRAIN_MODE:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self):
        self.model.save(MODEL_FILE)
        with open(REPLAY_FILE, "wb") as f:
            pickle.dump(self.memory, f)
        print("Model + replay buffer saved")

# =========================
# HUD STABILIZER
# =========================
class HUDStabilizer:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.last_valid = None

    def _is_sane(self, hud):
        if self.last_valid is None:
            return True
        if abs(hud["speed"] - self.last_valid["speed"]) > 300:
            return False
        p = hud["position"]
        lp = self.last_valid["position"]
        if any(abs(p[i] - lp[i]) > 200 for i in range(3)):
            return False
        if hud["checkpoint_current"] < self.last_valid["checkpoint_current"]:
            return False
        return True

    def update(self, raw_hud):
        if not self._is_sane(raw_hud):
            return self.last_valid or raw_hud

        self.buffer.append(raw_hud)

        def median(key, idx=None):
            vals = []
            for h in self.buffer:
                v = h[key]
                if idx is not None:
                    vals.append(v[idx])
                else:
                    vals.append(v)
            return float(np.median(vals))

        stabilized = {
            "position": (median("position",0), median("position",1), median("position",2)),
            "velocity": raw_hud["velocity"],
            "yaw": median("yaw"),
            "pitch": median("pitch"),
            "roll": median("roll"),
            "speed": median("speed"),
            "steer_direction": raw_hud["steer_direction"],
            "turning_rate": raw_hud["turning_rate"],
            "checkpoint_current": int(median("checkpoint_current")),
            "checkpoint_total": FORCED_CHECKPOINT_TOTAL
        }

        self.last_valid = stabilized
        return stabilized

# =========================
# MAIN LOOP
# =========================
agent = DQNAgent()
stabilizer = HUDStabilizer()
episode = 0

window = get_window()  # Get the game window once

while True:
    episode += 1
    episode_memory = []

    print(f"\n--- EPISODE {episode} ---")

    prev_hud = read_hud(window)
    prev_hud = stabilizer.update(prev_hud)

    stuck_start = None
    race_active = True
    checkpoint_buffer = deque(maxlen=50)  # For 10-frame confirmation

    while race_active:
        # ------------------------------
        # Read & stabilize HUD
        # ------------------------------
        raw_hud = read_hud(window)
        hud = stabilizer.update(raw_hud)

        # ------------------------------
        # FALL DETECTION
        # ------------------------------
        if FALL_RESET_ENABLED and hud.get("position") and hud["position"][1] < FALL_Z_THRESHOLD:
            print(f"--- EPISODE FALL DETECTED (Y={hud['position'][1]:.2f}) — Restarting race ---")
            keyboard_controller.press(Key.enter)
            time.sleep(0.2)
            keyboard_controller.release(Key.enter)
            break

        # ------------------------------
        # Compute state & select action
        # ------------------------------
        state = hud_to_state(hud)
        action = agent.act(state)
        press_action(action)

        # ------------------------------
        # Update checkpoint buffer
        # ------------------------------
        checkpoint_buffer.append(hud["checkpoint_current"])
        done = False
        if len(checkpoint_buffer) == 50 and all(x == hud["checkpoint_total"] for x in checkpoint_buffer):
            done = True

        # ------------------------------
        # Debug print (commented out)
        # ------------------------------
#        print(f"Speed: {hud.get('speed',0.0):.2f} | Pos: {hud.get('position')} | Checkpoint: {hud.get('checkpoint_current')}/{hud.get('checkpoint_total')}")

        time.sleep(ACTION_DELAY)

        # ------------------------------
        # Next state
        # ------------------------------
        next_hud = read_hud(window)
        next_hud = stabilizer.update(next_hud)
        next_state = hud_to_state(next_hud)
        reward = compute_reward(prev_hud, next_hud, done)
        episode_memory.append((state, action, reward, next_state, done))
        prev_hud = next_hud

        # ------------------------------
        # Stuck detection
        # ------------------------------
        if hud.get("speed",0.0) < STUCK_SPEED_THRESHOLD:
            if stuck_start is None:
                stuck_start = time.time()
            elif time.time() - stuck_start > STUCK_TIME_LIMIT:
                print("--- AI stuck for too long — restarting race ---")
                keyboard_controller.press(Key.enter)
                time.sleep(0.2)
                keyboard_controller.release(Key.enter)
                break
        else:
            stuck_start = None

        # ------------------------------
        # Check if finished
        # ------------------------------
        if done:
            print("Race finished!")
            race_active = False

    # ------------------------------
    # Replay / Training
    # ------------------------------
    if TRAIN_MODE:
        for exp in episode_memory:
            agent.remember(exp)

        for _ in range(10):
            agent.replay()

        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target()

        agent.save()
