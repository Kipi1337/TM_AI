#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trackmania AI Driver – Evolutionary Strategy with Elite Pool

Author:  <your name>
Date:    2026-01-27
"""

import cv2
import json
import mss
import numpy as np
import os
import pyautogui
import pygetwindow as gw
import pytesseract
import random
import re
import time
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# --------------------------------------------------------------------------- #
#  Configuration & Constants                                                  #
# --------------------------------------------------------------------------- #

WINDOW_TITLE = (
    "Trackmania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.10)"
)

HUD_CROP = (15, 60, 345, 285)

WAYPOINTS = [
    (243.983, 88.014, 687.997),
    (281.946, 88.014, 741.501),
    (407.122, 88.014, 749.829),
    (444.749, 88.014, 695.584),
    (970.589, 24.014, 673.361),
    (971.381, 24.014, 209.084),
    (639.263, 24.013, 176.736),
    (22.334, 113.359, 176.363),
]

KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ENTER = "enter"

INPUT_KEYS = [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]

TIME_BETWEEN_INPUTS = 0.05
EPS_SPEED_START = 0.03

ELITE_POOL_SIZE = 50
ELITE_FILE = "elite_pool.json"


# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #

def get_window() -> gw.Window:
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        raise RuntimeError(f"Window with title '{WINDOW_TITLE}' not found.")
    return windows[0]


def capture_screenshot(window: gw.Window) -> Image.Image:
    with mss.mss() as sct:
        img = sct.grab(
            (window.left, window.top, window.left + window.width, window.top + window.height)
        )
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
    except ValueError:
        return []


# --------------------------------------------------------------------------- #
#  Episode                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Episode:
    actions: List[str]
    finish_time: Optional[float]
    checkpoint: int
    signature: str = field(init=False)

    def __post_init__(self):
        # behavioral fingerprint (first 50 actions)
        self.signature = "".join(self.actions[:50])


# --------------------------------------------------------------------------- #
#  AI                                                                         #
# --------------------------------------------------------------------------- #

class TrackmaniaAI:
    def __init__(self, learning: bool = True):
        self.window = get_window()
        self.learning = learning
        self.elite_pool: List[Episode] = []
        self.load_elites()

    # ------------------------------------------------------------------ #
    # HUD                                                                #
    # ------------------------------------------------------------------ #

    def _ocr_hud(self) -> dict:
        img = capture_screenshot(self.window).crop(HUD_CROP)
        txt = pytesseract.image_to_string(
            preprocess_for_ocr(img), lang="eng", config="--psm 6"
        )
        nums = extract_numbers(txt)

        return {
            "speed": nums[12] if len(nums) > 12 else 0.0,
            "checkpoint": int(nums[16]) if len(nums) > 16 else 0,
        }

    # ------------------------------------------------------------------ #
    # Evolution                                                           #
    # ------------------------------------------------------------------ #

    def _mutate(self, actions: List[str]) -> List[str]:
        new = actions.copy()

        if random.random() < 0.3:
            new.insert(random.randint(0, len(new)), random.choice(INPUT_KEYS))
        if new and random.random() < 0.2:
            del new[random.randint(0, len(new) - 1)]
        if new and random.random() < 0.4:
            new[random.randint(0, len(new) - 1)] = random.choice(INPUT_KEYS)

        return new

    def _select_parent(self) -> List[str]:
        if not self.elite_pool:
            return [random.choice(INPUT_KEYS) for _ in range(200)]

        # weighted random: better checkpoints & time → more likely
        weights = [
            (ep.checkpoint + 1) / (ep.finish_time or 999)
            for ep in self.elite_pool
        ]
        parent = random.choices(self.elite_pool, weights=weights, k=1)[0]
        return parent.actions

    # ------------------------------------------------------------------ #
    # Elite Pool                                                          #
    # ------------------------------------------------------------------ #

    def _try_add_elite(self, episode: Episode):
        # reject duplicates
        if any(ep.signature == episode.signature for ep in self.elite_pool):
            return

        self.elite_pool.append(episode)

        # sort by checkpoint desc, time asc
        self.elite_pool.sort(
            key=lambda e: (-e.checkpoint, e.finish_time or float("inf"))
        )

        if len(self.elite_pool) > ELITE_POOL_SIZE:
            self.elite_pool.pop(-1)

        self.save_elites()

    def load_elites(self):
        if not os.path.exists(ELITE_FILE):
            return
        with open(ELITE_FILE, "r") as f:
            raw = json.load(f)
        for e in raw:
            self.elite_pool.append(Episode(**e))

    def save_elites(self):
        with open(ELITE_FILE, "w") as f:
            json.dump(
                [
                    {
                        "actions": e.actions,
                        "finish_time": e.finish_time,
                        "checkpoint": e.checkpoint,
                    }
                    for e in self.elite_pool
                ],
                f,
                indent=2,
            )

# ------------------------------------------------------------------ #
# Main Loop                                                           #
# ------------------------------------------------------------------ #

    def run(self):
        print("=== Trackmania AI (Elite Pool) ===")
    
        while True:
            # ---------- 1️⃣ Pick / mutate a sequence of actions ----------
            actions = self._mutate(self._select_parent())
            start_time = None
            checkpoint = 0
    
            # Keep track of the key that is currently pressed so we can release it later
            current_key: Optional[str] = None
    
            for act in actions:
                # ---------- 2️⃣ Log the action ----------
                print(f"[{time.strftime('%H:%M:%S')}] Sending key: {act}")
    
                if self.learning:
                    # If we were already holding a key, release it first
                    if current_key is not None and current_key != act:
                        pyautogui.keyUp(current_key)
    
                    # Hold the new key down
                    pyautogui.keyDown(act)
                    current_key = act
    
                    # Small pause to give the game time to react
                    time.sleep(TIME_BETWEEN_INPUTS)
    
                # ---------- 3️⃣ Read HUD ----------
                hud = self._ocr_hud()
                checkpoint = max(checkpoint, hud["checkpoint"])
    
                if start_time is None and hud["speed"] > EPS_SPEED_START:
                    start_time = time.time()
    
            # ---------- 4️⃣ Release the last key that was held ----------
            if self.learning and current_key is not None:
                pyautogui.keyUp(current_key)
    
            finish_time = (
                time.time() - start_time if start_time is not None else None
            )
    
            ep = Episode(actions, finish_time, checkpoint)
            self._try_add_elite(ep)
    
            print(
                f"Run finished | time={finish_time} | checkpoint={checkpoint} "
                f"| elites={len(self.elite_pool)}"
            )
    
            if self.learning:
                pyautogui.press(KEY_ENTER)
                time.sleep(1.5)



# --------------------------------------------------------------------------- #
#  Entry                                                                      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    TrackmaniaAI(learning=True).run()
