#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trackmania AI Driver – Evolutionary Strategy with full HUD output

Author:  <your name>
Date:    2026‑01‑26
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import mss
import numpy as np
import pyautogui
import pytesseract
import pygetwindow as gw
from PIL import Image


# --------------------------------------------------------------------------- #
#  Configuration & Constants                                                 #
# --------------------------------------------------------------------------- #

WINDOW_TITLE = (
    "Trackmania Modded Forever (2.12.0) [default]: "
    "TMInterface (2.2.1), CoreMod (1.0.10)"
)

HUD_CROP: Tuple[int, int, int, int] = (15, 60, 345, 285)
WAYPOINTS: List[Tuple[float, float, float]] = [
    (243.983, 88.014, 687.997),   # Start
    (281.946, 88.014, 741.501),   # Turn 1
    (407.122, 88.014, 749.829),   # Turn 2
    (444.749, 88.014, 695.584),   # Turn 3
    (970.589, 24.014, 673.361),   # Turn 4
    (971.381, 24.014, 209.084),   # Turn 5
    (639.263, 24.013, 176.736),   # Turn 6
    (22.334, 113.359, 176.363),   # Finish
]
FINISH_COORDS = WAYPOINTS[-1]

KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ENTER = "enter"

INPUT_KEYS = [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]
TIME_BETWEEN_INPUTS = 0.05          # seconds

BEST_EPISODE_FILE = "best_episode.json"


# --------------------------------------------------------------------------- #
#  Helper functions                                                          #
# --------------------------------------------------------------------------- #

def get_window() -> gw.Window:
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        raise RuntimeError(f"Window with title '{WINDOW_TITLE}' not found.")
    return windows[0]


def capture_screenshot(window: gw.Window) -> Image.Image:
    left, top, width, height = (
        window.left,
        window.top,
        window.width,
        window.height,
    )
    with mss.mss() as sct:
        img = sct.grab((left, top, left + width, top + height))
    return Image.frombytes("RGB", img.size, img.rgb)


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    np_img = np.array(gray)
    _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def extract_numbers(text: str) -> List[float]:
    nums = re.findall(r"-?\d+\.\d+|-\d+|\d+", text)
    try:
        return [float(n) for n in nums]
    except ValueError:  # pragma: no cover
        return []


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


# --------------------------------------------------------------------------- #
#  Episode – a single race run                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Episode:
    actions: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    def add(self, action: str):
        self.actions.append(action)

    def start(self):
        self.start_time = time.time()

    def finish(self):
        if self.start_time is not None:
            self.finish_time = time.time() - self.start_time


# --------------------------------------------------------------------------- #
#  The main AI class                                                       #
# --------------------------------------------------------------------------- #

class TrackmaniaAI:
    def __init__(self, learning: bool = True) -> None:
        self.window = get_window()
        self.learning_mode = learning

        self.current_episode = Episode()

        # Best episode that survived a race (lowest finish time)
        self.best_episode: Optional[Episode] = None
        self.load_best_episode()

        # Helper counters for stall detection
        self.stall_counter = 0
        self.race_started = False
        self.last_speed = 0.0

    # --------------------------------------------------------------------- #
    #  HUD monitoring                                                        #
    # --------------------------------------------------------------------- #

    def _ocr_hud(self) -> str:
        screenshot = capture_screenshot(self.window)
        hud_img = screenshot.crop(HUD_CROP)
        processed = preprocess_for_ocr(hud_img)
        return pytesseract.image_to_string(processed, lang="eng", config="--psm 6")

    def _parse_hud(self, text: str) -> dict:
        """
        Keep *all* numbers that the OCR can find.  
        The original HUD shows 14 numeric values in this exact order:

            pos(3), vel(3), yaw, pitch, roll,
            speed, steer_dir, turning_rate,
            checkpoint_current, checkpoint_total
        """
        nums = extract_numbers(text)

        data = {
            "position": tuple(nums[0:3]) if len(nums) >= 3 else None,
            "velocity": tuple(nums[3:6]) if len(nums) >= 6 else None,
            "yaw": nums[6] if len(nums) >= 7 else None,
            "pitch": nums[7] if len(nums) >= 8 else None,
            "roll": nums[8] if len(nums) >= 9 else None,
            "speed": nums[9] if len(nums) >= 10 else None,
            "steer_direction": nums[10] if len(nums) >= 11 else None,
            "turning_rate": nums[11] if len(nums) >= 12 else None,
            "checkpoint_current": int(nums[12]) if len(nums) >= 13 else None,
            "checkpoint_total": int(nums[13]) if len(nums) >= 14 else None,
        }
        return data

    # --------------------------------------------------------------------- #
    #  Navigation logic – simple look‑ahead to the next waypoint          #
    # --------------------------------------------------------------------- #

    def _next_waypoint(self, current_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if current_pos is None:
            return WAYPOINTS[0]
        for wp in WAYPOINTS:
            if distance(current_pos, wp) > 10.0:
                return wp
        return WAYPOINTS[-1]

    def _steering_direction(self, current_pos: Tuple[float, float, float]) -> str:
        if current_pos is None:
            return KEY_RIGHT

        wp = self._next_waypoint(current_pos)
        dx, _, dz = wp[0] - current_pos[0], 0.0, wp[2] - current_pos[2]
        heading = np.array([0.0, 0.0, 1.0])          # forward (+Z) – crude
        cross_z = heading[0] * dz - heading[2] * dx
        return KEY_LEFT if cross_z > 0 else KEY_RIGHT

    # --------------------------------------------------------------------- #
    #  Input sending                                                        #
    # --------------------------------------------------------------------- #

    def _send_input(self, key: str):
        pyautogui.keyDown(key)
        time.sleep(TIME_BETWEEN_INPUTS)
        pyautogui.keyUp(key)

    # --------------------------------------------------------------------- #
    #  Evolutionary strategy – mutation                                    #
    # --------------------------------------------------------------------- #

    def _mutate(self, actions: List[str]) -> List[str]:
        new_actions = actions.copy()

        if random.random() < 0.3:
            pos = random.randint(0, len(new_actions))
            new_actions.insert(pos, random.choice(INPUT_KEYS))

        if new_actions and random.random() < 0.2:
            del_pos = random.randint(0, len(new_actions) - 1)
            del new_actions[del_pos]

        if new_actions and random.random() < 0.4:
            repl_pos = random.randint(0, len(new_actions) - 1)
            new_actions[repl_pos] = random.choice(INPUT_KEYS)

        return new_actions

    def _generate_episode(self) -> List[str]:
        if not self.learning_mode:
            return self.best_episode.actions if self.best_episode else []

        if self.best_episode is None:
            # First run – start with a random sequence of 200 actions
            return [random.choice(INPUT_KEYS) for _ in range(200)]

        mutated = self._mutate(self.best_episode.actions)
        if len(mutated) < 50:
            mutated += [random.choice(INPUT_KEYS) for _ in range(200 - len(mutated))]
        return mutated

    # --------------------------------------------------------------------- #
    #  Persistence – read/write best episode                               #
    # --------------------------------------------------------------------- #

    def load_best_episode(self):
        if os.path.exists(BEST_EPISODE_FILE):
            with open(BEST_EPISODE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.best_episode = Episode(
                    actions=data.get("actions", []),
                    finish_time=data.get("finish_time"),
                )
        else:
            self.best_episode = None

    def save_best_episode(self):
        if self.best_episode is None:
            return
        data = {
            "actions": self.best_episode.actions,
            "finish_time": self.best_episode.finish_time,
        }
        with open(BEST_EPISODE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # --------------------------------------------------------------------- #
    #  Main loop – orchestrates everything                                 #
    # --------------------------------------------------------------------- #

    def run(self):
        print("=== Trackmania AI Driver ===")
        mode = "Learning" if self.learning_mode else "Watch"
        print(f"Mode: {mode}")
        print(f"Best episode file: {BEST_EPISODE_FILE}\n")

        try:
            while True:
                actions = self._generate_episode()
                if not actions:
                    time.sleep(5)
                    continue

                # Reset per‑episode state
                self.current_episode = Episode(actions=actions.copy())
                self.current_episode.start()
                self.stall_counter = 0
                self.race_started = False
                self.last_speed = 0.0

                for act in actions:
                    if self.learning_mode:
                        self._send_input(act)

                    # Every 20th action we read the HUD so that we can
                    # decide whether to stop early (e.g. crash)
                    if len(self.current_episode.actions) % 20 == 0:
                        hud_text = self._ocr_hud()
                        hud_data = self._parse_hud(hud_text)

                        pos = hud_data["position"]
                        speed = hud_data["speed"] or 0.0

                        # --- race timer logic ---------------------------------
                        if not self.race_started and speed > 0.03:
                            self.race_started = True
                            print("\n--- RACE STARTED! ---")

                        # --- stall detection ----------------------------------
                        if speed < 0.01:
                            self.stall_counter += 1
                            if self.stall_counter >= 5:      # abort after 5 frames
                                print("Car stalled – aborting episode")
                                break
                        else:
                            self.stall_counter = 0

                        # --- status line ------------------------------------
                        elapsed = time.time() - self.current_episode.start_time
                        steering = (
                            "FWD"
                            if act == KEY_UP
                            else ("BWD" if act == KEY_DOWN else self._steering_direction(pos))
                        )
                        print(
                            f"[{elapsed:6.2f}s] Pos:{pos} "
                            f"Vel:{hud_data['velocity']} "
                            f"Yaw/Pitch/Roll:{hud_data['yaw']}/{hud_data['pitch']}/{hud_data['roll']} "
                            f"Spd:{speed:5.1f}  "
                            f"SteerDir:{hud_data['steer_direction']}  TurnRate:{hud_data['turning_rate']} "
                            f"Ckpt:{hud_data['checkpoint_current']}/{hud_data['checkpoint_total']}"
                        )

                # Finish episode
                self.current_episode.finish()
                finish_time = (
                    self.current_episode.finish_time
                    if self.current_episode.finish_time is not None
                    else float("inf")
                )
                print(f"\nEpisode finished in {finish_time:.2f} s")

                # Update best episode if this one was better
                if (
                    self.best_episode is None
                    or finish_time < (self.best_episode.finish_time or float("inf"))
                ):
                    self.best_episode = Episode(
                        actions=actions, finish_time=finish_time
                    )
                    print(f"NEW BEST! {finish_time:.2f} s")
                    self.save_best_episode()

                # Reset the race (if we are in learning mode)
                if self.learning_mode:
                    pyautogui.press(KEY_ENTER)   # press Enter to restart
                    time.sleep(1.5)

                print("-" * 40)
                time.sleep(2)

        except KeyboardInterrupt:
            print("\nInterrupted – exiting.")
        finally:
            self.save_best_episode()


# --------------------------------------------------------------------------- #
#  Entry point                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ai = TrackmaniaAI(learning=True)   # set to False for watch‑only
    ai.run()
