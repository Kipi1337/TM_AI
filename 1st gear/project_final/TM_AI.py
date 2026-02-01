import os
import time
import random
import pickle
import numpy as np
from collections import deque
from pynput.keyboard import Key, Controller, Listener
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
from dataclasses import dataclass

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    SPEED_REWARD_MULT: float = 0.20
    CHECKPOINT_REWARD: float = 1500.0
    FINISH_REWARD: float = 10000.0
    BRAKE_PENALTY: float = -8.0
    OSCILLATION_PENALTY: float = -15.0
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.05
    EPSILON_DECAY: float = 0.995
    GAMMA: float = 0.99
    LEARNING_RATE: float = 0.0003
    BATCH_SIZE: int = 128
    TARGET_UPDATE: int = 5
    MAX_MEMORY: int = 200_000
    FINISH_FRAMES_REQUIRED: int = 8
    CHECKPOINT_FRAMES_REQUIRED: int = 4
    MIN_CHECKPOINT_DISTANCE: float = 30.0
    FALL_Z_THRESHOLD: float = 15.0
    FALL_FRAMES_REQUIRED: int = 15  # NEW: Must be below threshold for 15 consecutive frames
    FALL_PENALTY: float = -100.0
    ACTION_DELAY: float = 0.15

CONFIG = Config()
STATE_SIZE = 11
ACTION_SIZE = 7
REPLAY_FILE = "replay_buffer_robust.pkl"
MODEL_FILE = "dqn_model_robust.keras"
WINDOW_TITLE = "Trackmania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.10)"
HUD_CROP = (15, 60, 345, 285)
FORCED_CHECKPOINT_TOTAL = 3

keyboard_controller = Controller()
STOP_REQUESTED = False  # Emergency ESC exit flag

# =========================
# OUTLIER BUFFER
# =========================
class OutlierBuffer:
    def __init__(self, size=10, threshold=100.0):
        self.buffer = deque(maxlen=size)
        self.threshold = threshold
        
    def filter(self, value):
        if len(self.buffer) < 3:
            self.buffer.append(value)
            return value
            
        median = np.median(list(self.buffer))
        
        if abs(value - median) > self.threshold:
            print(f"  ⚠️  Outlier rejected: {value:.1f} (median: {median:.1f})")
            return median
            
        self.buffer.append(value)
        return value
        
    def reset(self):
        self.buffer.clear()

# =========================
# ACTIONS
# =========================
def press_action(action):
    # Release all first
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
# RACE RESTART SEQUENCE
# =========================
def restart_race_after_finish():
    """Wait 4 seconds then press Enter 3 times to restart the race"""
    print("⏳ Race finished! Waiting 4 seconds for results...")
    time.sleep(4)
    
    print("🏁 Restarting race (Enter x3)...")
    for i in range(3):
        keyboard_controller.press(Key.enter)
        time.sleep(0.2)
        keyboard_controller.release(Key.enter)
        if i < 2:  # Small delay between presses (not after last)
            time.sleep(0.3)
    print("✅ Race restarted")

# =========================
# HUD FUNCTIONS
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

def read_hud(window, z_buffer=None):
    screenshot = capture_screenshot(window)
    hud_img = screenshot.crop(HUD_CROP)
    processed = preprocess_for_ocr(hud_img)
    text = pytesseract.image_to_string(processed, lang="eng", config="--psm 6")
    nums = extract_numbers(text)
    
    if len(nums) >= 3:
        # Trackmania HUD order: X (0), Z-height (1), Y-forward (2)
        x = nums[0]
        height = nums[1]  # This is the Z coordinate (up/down)
        y_forward = nums[2]  # This is Y (forward/back)
        
        # Apply outlier filtering ONLY to height (index 1)
        if z_buffer:
            height = z_buffer.filter(height)
            
        position = (x, height, y_forward)  # position[1] is now filtered height
    else:
        position = (0, 100, 0)  # Safe default: high up
    
    data = {
        "position": position,  # Fall detection checks position[1]
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
    
    return data

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
# REWARD
# =========================
def compute_reward(prev_hud, hud, finished, oscillating=False, checkpoint_reached=False, action=6):
    reward = 0.0
    speed = hud.get("speed", 0.0)
    prev_speed = prev_hud.get("speed", 0.0)

    reward += speed * CONFIG.SPEED_REWARD_MULT
    reward += max(0, speed - prev_speed) * 0.1

    if checkpoint_reached:
        reward += CONFIG.CHECKPOINT_REWARD
        print(f"🚩 CHECKPOINT +{CONFIG.CHECKPOINT_REWARD}")

    if action == 1:
        reward += CONFIG.BRAKE_PENALTY
        
    if oscillating:
        reward += CONFIG.OSCILLATION_PENALTY

    if speed < 1.0:
        reward -= 2.0

    if prev_speed > 5 and speed < 0.5:
        reward -= 50.0
        
    if finished:
        reward += CONFIG.FINISH_REWARD

    return reward

# =========================
# VALIDATORS (Including new FallValidator)
# =========================
class FallValidator:
    """Requires 15 consecutive frames below threshold to confirm fall"""
    def __init__(self):
        self.consecutive_fall_frames = 0
        self.fall_confirmed = False
        
    def update(self, hud):
        """Returns True only after 15 consecutive frames below threshold"""
        if self.fall_confirmed:
            return True
            
        z_height = hud.get("position", (0, 100, 0))[1]
        
        if z_height < CONFIG.FALL_Z_THRESHOLD:
            self.consecutive_fall_frames += 1
            if self.consecutive_fall_frames >= CONFIG.FALL_FRAMES_REQUIRED:
                print(f"💥 Fall confirmed after {self.consecutive_fall_frames} consecutive frames (Z={z_height:.2f})")
                self.fall_confirmed = True
                return True
        else:
            # Reset counter if we get a frame above threshold (not falling)
            if self.consecutive_fall_frames > 0:
                print(f"  ↩️  Fall detection reset (Z={z_height:.2f}, had {self.consecutive_fall_frames} frames)")
            self.consecutive_fall_frames = 0
            
        return False
        
    def reset(self):
        self.consecutive_fall_frames = 0
        self.fall_confirmed = False

class FinishValidator:
    def __init__(self):
        self.frames_at_finish = 0
        self.finish_pos = None
        
    def update(self, hud, checkpoint_confirmed):
        if checkpoint_confirmed >= FORCED_CHECKPOINT_TOTAL and hud["speed"] < 1.0:
            if self.finish_pos is None:
                self.finish_pos = hud["position"]
            if self.finish_pos:
                dist = np.linalg.norm(np.array(hud["position"]) - np.array(self.finish_pos))
                if dist < 5.0:
                    self.frames_at_finish += 1
                else:
                    self.frames_at_finish = 0
                    
        finished = self.frames_at_finish >= CONFIG.FINISH_FRAMES_REQUIRED
        if finished and self.frames_at_finish == CONFIG.FINISH_FRAMES_REQUIRED:
            print("🏆 FINISH CONFIRMED!")
        return finished
        
    def reset(self):
        self.frames_at_finish = 0
        self.finish_pos = None

class CheckpointValidator:
    def __init__(self):
        self.history = deque(maxlen=5)
        self.confirmed = 0
        self.last_checkpoint_pos = None
        
    def update(self, raw_value, current_pos):
        self.history.append(raw_value)
        if len(self.history) < CONFIG.CHECKPOINT_FRAMES_REQUIRED:
            return self.confirmed, False
            
        recent = list(self.history)[-CONFIG.CHECKPOINT_FRAMES_REQUIRED:]
        if len(set(recent)) != 1:
            return self.confirmed, False
            
        candidate = recent[0]
        
        if candidate > self.confirmed:
            if candidate != self.confirmed + 1:
                return self.confirmed, False
            
            if self.last_checkpoint_pos is not None:
                dist_moved = np.linalg.norm(
                    np.array(current_pos) - np.array(self.last_checkpoint_pos)
                )
                if self.confirmed > 0 and dist_moved < CONFIG.MIN_CHECKPOINT_DISTANCE:
                    return self.confirmed, False
            
            self.confirmed = candidate
            self.last_checkpoint_pos = current_pos
            return self.confirmed, True
            
        return self.confirmed, False
        
    def reset(self):
        self.history.clear()
        self.confirmed = 0
        self.last_checkpoint_pos = None

# =========================
# DQN AGENT
# =========================
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=CONFIG.MAX_MEMORY)
        self.gamma = CONFIG.GAMMA
        self.epsilon = CONFIG.EPSILON_START
        self.epsilon_min = CONFIG.EPSILON_MIN
        self.epsilon_decay = CONFIG.EPSILON_DECAY
        self.lr = CONFIG.LEARNING_RATE

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
            Dense(256, activation="relu", input_shape=(STATE_SIZE,)),
            Dense(256, activation="relu"),
            Dense(ACTION_SIZE, activation="linear")
        ])
        model.compile(optimizer=Adam(self.lr), loss="mse")
        return model

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def remember(self, exp):
        self.memory.append(exp)

    def replay(self):
        if len(self.memory) < CONFIG.BATCH_SIZE:
            return

        batch = random.sample(self.memory, CONFIG.BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(
                    self.target_model.predict(next_state, verbose=0)[0]
                )
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self):
        self.model.save(MODEL_FILE)
        with open(REPLAY_FILE, "wb") as f:
            pickle.dump(self.memory, f)

# =========================
# EMERGENCY ESC LISTENER
# =========================
def on_key_press(key):
    global STOP_REQUESTED
    try:
        if key == Key.esc:
            print("\n🛑 EMERGENCY STOP (ESC pressed)")
            STOP_REQUESTED = True
            for k in [Key.up, Key.down, Key.left, Key.right]:
                try:
                    keyboard_controller.release(k)
                except:
                    pass
            return False  # Stops listener
    except:
        pass

# =========================
# MAIN LOOP (Fixed)
# =========================
def main():
    global STOP_REQUESTED
    print("🏎️  Trackmania AI Starting...")
    print(f"Checkpoint Reward: {CONFIG.CHECKPOINT_REWARD}")
    print(f"Fall detection: {CONFIG.FALL_FRAMES_REQUIRED} consecutive frames below Z={CONFIG.FALL_Z_THRESHOLD}")

    agent = DQNAgent()
    finish_validator = FinishValidator()
    checkpoint_validator = CheckpointValidator()
    fall_validator = FallValidator()
    z_buffer = OutlierBuffer(size=10, threshold=400.0)

    listener = Listener(on_press=on_key_press)
    listener.start()

    episode = 0
    window = get_window()

    try:
        while True:
            if STOP_REQUESTED:
                raise KeyboardInterrupt

            episode += 1
            print(f"\n🏁 EPISODE {episode} (ε={agent.epsilon:.3f})")
            
            finish_validator.reset()
            checkpoint_validator.reset()
            fall_validator.reset()
            z_buffer.reset()
            
            # Robust initial HUD read with retry
            init_retries = 0
            raw_hud = None
            while init_retries < 30:  # Try for 3 seconds (30 * 0.1)
                try:
                    raw_hud = read_hud(window, z_buffer)
                    # Verify car is not falling (Z should be > threshold)
                    if raw_hud["position"][1] > CONFIG.FALL_Z_THRESHOLD:
                        break
                    else:
                        print(f"  ⏳ Waiting for car to respawn... (Z={raw_hud['position'][1]:.1f})")
                        time.sleep(0.1)
                        init_retries += 1
                except Exception as e:
                    print(f"  ⏳ Waiting for valid HUD... ({e})")
                    time.sleep(0.1)
                    init_retries += 1
            
            if raw_hud is None:
                print("❌ Failed to read initial HUD, skipping episode")
                continue
                
            checkpoint_validator.last_checkpoint_pos = raw_hud["position"]
            prev_hud = raw_hud
            
            race_active = True
            episode_memory = []
            finished = False

            while race_active:
                if STOP_REQUESTED:
                    raise KeyboardInterrupt

                hud = read_hud(window, z_buffer)
                confirmed_cp, new_checkpoint = checkpoint_validator.update(
                    hud["checkpoint_current"], hud["position"]
                )
                hud["checkpoint_current"] = confirmed_cp
                finished = finish_validator.update(hud, confirmed_cp)
                
                # ROBUST FALL DETECTION
                fell = fall_validator.update(hud)
                
                if fell:
                    print(f"--- EPISODE FALL CONFIRMED — Restarting race ---")
                    
                    # CRITICAL FIX 1: Release all keys before restart
                    print("  🛑 Releasing all keys...")
                    press_action(6)  # Action 6 = no input (releases all)
                    time.sleep(0.1)  # Brief delay to ensure keys are released
                    
                    # Store experience
                    state = hud_to_state(prev_hud)
                    next_state = hud_to_state(hud)
                    agent.remember((state, 6, CONFIG.FALL_PENALTY, next_state, True))
                    
                    # Press enter to restart
                    keyboard_controller.press(Key.enter)
                    time.sleep(0.2)
                    keyboard_controller.release(Key.enter)
                    
                    finished = False
                    
                    # CRITICAL FIX 2: Wait for respawn before ending episode
                    print("  ⏳ Waiting for race to restart...")
                    time.sleep(2.0)  # Give game time to load
                    break

                state = hud_to_state(hud)
                action = agent.act(state)
                oscillating = False
                if len([m for m in episode_memory[-6:] if m[1] in [2,3]]) >= 4:
                    oscillating = True
                    
                press_action(action)
                time.sleep(CONFIG.ACTION_DELAY)

                next_hud = read_hud(window, z_buffer)
                next_state = hud_to_state(next_hud)
                
                reward = compute_reward(prev_hud, next_hud, finished, oscillating, new_checkpoint, action)
                episode_memory.append((state, action, reward, next_state, finished))
                prev_hud = next_hud

                if finished:
                    print("✅ Race finished successfully!")
                    race_active = False

            # After race ends (finish or fall)
            if finished and not STOP_REQUESTED:
                restart_race_after_finish()
            elif not finished and not STOP_REQUESTED:
                # Fall occurred - already waited 2s above, but ensure clean state
                print("  🧹 Cleaning up after fall...")
                press_action(6)  # Ensure no keys stuck
                time.sleep(0.5)

            # TRAINING
            print("Training...")
            for exp in episode_memory:
                agent.remember(exp)
            for _ in range(10):
                agent.replay()
            if episode % CONFIG.TARGET_UPDATE == 0:
                agent.update_target()
            agent.save()

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        # Release all keys on exit
        press_action(6)
        agent.save()
        print("Saved and exited")

if __name__ == "__main__":
    main()