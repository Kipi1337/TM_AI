#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trackmania AI Driver – Evolutionary Strategy with full HUD output

Author:  <your name>
Date:    2026‑01‑26

This script implements a very simple evolutionary driver for the
*Trackmania Modded Forever* game.  
The agent reads the in‑game HUD (via OCR), decides on steering
commands, and sends key presses to the game window.  The “learning”
mode mutates the best episode found so far; the “watch” mode simply
replays the stored best episode.

The code is intentionally kept straightforward – it should serve as a
starting point for anyone who wants to experiment with RL / evolutionary
algorithms in an interactive environment.
"""

# --------------------------------------------------------------------------- #
#  Imports                                                                   #
# --------------------------------------------------------------------------- #

import json          # For persistence of the best episode
import os            # File‑system utilities
import random        # Random number generation (mutation)
import re            # Regular expression for OCR parsing
import time          # Timing helpers

from dataclasses import dataclass, field   # Simple data container for episodes
from typing import List, Tuple, Optional    # Type hints for readability

# 3rd‑party libraries ------------------------------------------------------- #
import cv2           # OpenCV – image processing before OCR
import mss           # Screenshot capture of a single monitor
import numpy as np   # Numerical array manipulation
import pyautogui     # Simulate keyboard input to the game window
import pytesseract   # Optical Character Recognition (OCR)
import pygetwindow as gw  # Find the game window by title
from PIL import Image        # Pillow – image conversion helpers

# --------------------------------------------------------------------------- #
#  Configuration & Constants                                               #
# --------------------------------------------------------------------------- #

WINDOW_TITLE = (
    "Trackmania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.10)"
)

# Coordinates for the HUD region inside the game window.
HUD_CROP: Tuple[int, int, int, int] = (15, 60, 345, 285)

# Waypoints – a simplified “track” that the agent will try to follow.
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
FINISH_COORDS = WAYPOINTS[-1]       # Convenience alias

# Keys that the agent can press.
KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ENTER = "enter"

INPUT_KEYS = [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]
TIME_BETWEEN_INPUTS = 0.05          # Seconds to hold a key down

BEST_EPISODE_FILE = "best_episode.json"   # Persisted best episode


# --------------------------------------------------------------------------- #
#  Helper functions                                                        #
# --------------------------------------------------------------------------- #

def get_window() -> gw.Window:
    """
    Locate the Trackmania window by its title.
    Raises RuntimeError if not found – this is a fatal error.
    """
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        raise RuntimeError(f"Window with title '{WINDOW_TITLE}' not found.")
    return windows[0]


def capture_screenshot(window: gw.Window) -> Image.Image:
    """
    Grab the entire client area of *window* using mss and convert it to a
    Pillow image.  The function returns an RGB image that can be processed.
    """
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
    """
    Basic OCR pre‑processing:

      1. Convert to grayscale.
      2. Apply Otsu's thresholding to binarise the image.

    The result is a black‑and‑white image that Tesseract can read more
    reliably in this particular HUD layout.
    """
    gray = img.convert("L")
    np_img = np.array(gray)
    _, thresh = cv2.threshold(np_img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numeric tokens from *text* (the OCR result).  The regex
    matches signed integers and floats.  If conversion fails, an empty
    list is returned – the calling code will handle missing data.
    """
    nums = re.findall(r"-?\d+\.\d+|-\d+|\d+", text)
    try:
        return [float(n) for n in nums]
    except ValueError:  # pragma: no cover – defensive fallback
        return []


def distance(a: Tuple[float, float, float],
            b: Tuple[float, float, float]) -> float:
    """
    Euclidean distance in 3‑D space.  If either point is None we treat the
    distance as infinite so that the calling logic can safely ignore it.
    """
    if a is None or b is None:
        return float("inf")
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2) ** 0.5


# --------------------------------------------------------------------------- #
#  Episode – a single race run                                            #
# --------------------------------------------------------------------------- #

@dataclass
class Episode:
    """
    Simple container that stores the actions taken during an episode and
    timestamps for start / finish.  The dataclass auto‑generates __init__
    and __repr__ which makes debugging easier.
    """
    actions: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    def add(self, action: str) -> None:
        """Append an action to the episode."""
        self.actions.append(action)

    def start(self) -> None:
        """Mark the start time of the episode."""
        self.start_time = time.time()

    def finish(self) -> None:
        """
        Compute and store the elapsed time.  If `start_time` is missing
        (should not happen in normal flow) we simply leave `finish_time`
        unchanged.
        """
        if self.start_time is not None:
            self.finish_time = time.time() - self.start_time


# --------------------------------------------------------------------------- #
#  The main AI class                                                       #
# --------------------------------------------------------------------------- #

class TrackmaniaAI:
    """
    Core driver logic.  It handles:

      * Window detection & OCR
      * Simple navigation heuristics (next waypoint + steering)
      * Input simulation via pyautogui
      * Evolutionary mutation of the best episode
      * Persistence of the best episode to disk

    Parameters
    ----------
    learning : bool
        If True, the agent will mutate and improve its policy.
        If False it simply replays the stored best episode (watch mode).
    """
    def __init__(self, learning: bool = True) -> None:
        self.window = get_window()
        self.learning_mode = learning

        # Current race state
        self.current_episode = Episode()

        # Persisted best episode – loaded from disk or None initially.
        self.best_episode: Optional[Episode] = None
        self.load_best_episode()

        # Stall detection helpers
        self.stall_counter = 0
        self.race_started = False
        self.last_speed = 0.0

    # --------------------------------------------------------------------- #
    #  HUD monitoring                                                        #
    # --------------------------------------------------------------------- #

    def _ocr_hud(self) -> str:
        """
        Grab the HUD region of the game window and run OCR on it.
        Returns the raw string extracted by Tesseract.
        """
        screenshot = capture_screenshot(self.window)
        hud_img = screenshot.crop(HUD_CROP)
        processed = preprocess_for_ocr(hud_img)
        return pytesseract.image_to_string(processed,
                                          lang="eng",
                                          config="--psm 6")

    def _parse_hud(self, text: str) -> dict:
        """
        Convert the OCR string into a dictionary of numeric values.
        The HUD format is fixed – we know exactly which number appears
        at each position.  Missing data is represented by None.

        Returns keys such as 'position', 'velocity', 'speed',
        'steer_direction', etc.
        """
        # Parse all numbers in the OCR output
        nums = extract_numbers(text)

        data = {
            "position": tuple(nums[0:3]) if len(nums) >= 3 else None,
            "velocity": tuple(nums[3:6]) if len(nums) >= 6 else None,
            "yaw": nums[6] if len(nums) >= 7 else None,
            "pitch": nums[7] if len(nums) >= 8 else None,
            "roll": nums[8] if len(nums) >= 9 else None,
            "speed": nums[12] if len(nums) >= 13 else None,
            "steer_direction": nums[14] if len(nums) >= 15 else None,
            "turning_rate": nums[15] if len(nums) >= 16 else None,
            "checkpoint_current": int(nums[16]) if len(nums) >= 17 else None,
            "checkpoint_total": int(nums[17]) if len(nums) >= 18 else None,
        }

        return data

    # --------------------------------------------------------------------- #
    #  Navigation logic – simple look‑ahead to the next waypoint          #
    # --------------------------------------------------------------------- #

    def _next_waypoint(self, current_pos: Tuple[float, float, float]
                       ) -> Tuple[float, float, float]:
        """
        Determine which waypoint we should aim for next.

        If `current_pos` is None (e.g. OCR failed) we default to the
        first waypoint.  Otherwise we iterate through WAYPOINTS and return
        the first one that is more than 10 m away – this keeps the agent
        from “teleporting” to the final point too early.
        """
        if current_pos is None:
            return WAYPOINTS[0]
        for wp in WAYPOINTS:
            if distance(current_pos, wp) > 10.0:
                return wp
        return WAYPOINTS[-1]

    def _steering_direction(self,
                            current_pos: Tuple[float, float, float]) -> str:
        """
        Very naive steering logic:

          * Compute the vector to the next waypoint.
          * Use a cross‑product with an approximate forward direction
            (here we assume +Z is “forward”).
          * If the cross‑product’s Z component is positive we steer left,
            otherwise right.

        This does **not** use the car’s yaw; it simply biases steering
        based on relative position.
        """
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

    def _send_input(self, key: str) -> None:
        """
        Simulate a single key press.  We hold the key for
        TIME_BETWEEN_INPUTS seconds to make sure the game registers it.
        """
        pyautogui.keyDown(key)
        time.sleep(TIME_BETWEEN_INPUTS)
        pyautogui.keyUp(key)

    # --------------------------------------------------------------------- #
    #  Evolutionary strategy – mutation                                    #
    # --------------------------------------------------------------------- #

    def _mutate(self, actions: List[str]) -> List[str]:
        """
        Create a new action list by randomly inserting, deleting,
        or replacing commands in the parent sequence.

        Probabilities are hard‑coded but can be tuned for different
        exploration behaviours.
        """
        new_actions = actions.copy()

        # 30 % chance to insert a random key at a random position
        if random.random() < 0.3:
            pos = random.randint(0, len(new_actions))
            new_actions.insert(pos, random.choice(INPUT_KEYS))

        # 20 % chance to delete one command (if any exist)
        if new_actions and random.random() < 0.2:
            del_pos = random.randint(0, len(new_actions) - 1)
            del new_actions[del_pos]

        # 40 % chance to replace a single command
        if new_actions and random.random() < 0.4:
            repl_pos = random.randint(0, len(new_actions) - 1)
            new_actions[repl_pos] = random.choice(INPUT_KEYS)

        return new_actions

    def _generate_episode(self) -> List[str]:
        """
        Return a sequence of actions for the next run.

        * In learning mode we mutate the best episode found so far.
          If no best exists yet, we start with 200 random actions.
        * In watch‑only mode we simply replay the stored best episode
          (or an empty list if none is available).
        """
        if not self.learning_mode:
            return self.best_episode.actions if self.best_episode else []

        if self.best_episode is None:
            # First run – start with a random sequence of 200 actions
            return [random.choice(INPUT_KEYS) for _ in range(200)]

        mutated = self._mutate(self.best_episode.actions)
        if len(mutated) < 50:   # Guard against very short sequences
            mutated += [random.choice(INPUT_KEYS)
                        for _ in range(200 - len(mutated))]
        return mutated

    # --------------------------------------------------------------------- #
    #  Persistence – read/write best episode                               #
    # --------------------------------------------------------------------- #

    def load_best_episode(self) -> None:
        """
        Load the best episode from disk.  If the file does not exist
        we simply leave `self.best_episode` as None.
        """
        if os.path.exists(BEST_EPISODE_FILE):
            with open(BEST_EPISODE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.best_episode = Episode(
                    actions=data.get("actions", []),
                    finish_time=data.get("finish_time"),
                )
        else:
            self.best_episode = None

    def save_best_episode(self) -> None:
        """
        Persist the current best episode to disk.  The JSON contains
        only the action list and its finish time for simplicity.
        """
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

    def run(self) -> None:
        """
        The main driver loop.  It repeatedly:

          1. Generates a new episode (mutation or replay).
          2. Executes the actions, occasionally reading the HUD.
          3. Tracks race start / stall conditions.
          4. Updates the best episode if this run was faster.
          5. Restarts the game when in learning mode.

        The loop can be interrupted with Ctrl‑C – we then persist the
        best episode before exiting.
        """
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
                            else ("BWD" if act == KEY_DOWN else
                                  self._steering_direction(pos))
                        )
                        print(
                            f"[{elapsed:6.2f}s] Pos:{pos}"
                            f"Vel:{hud_data['velocity']}"
                            f"Yaw:{hud_data['yaw']}"
                            f"Pitch:{hud_data['pitch']}"
                            f"Roll:{hud_data['roll']}"
                            f"Spd:{speed:5.1f}"
                            f"SteerDir:{hud_data['steer_direction']}"
                            f"TurnRate:{hud_data['turning_rate']}"
                            f"Ckpt:{hud_data['checkpoint_current']}/"
                            f"{hud_data['checkpoint_total']}"
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
