#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full RL pipeline for a keyboard-controlled racing AI.
Author: ChatGPT (2026)

Added feature:
    • For the first 4 seconds of every race, the car will always accelerate (gas = True).
"""

import cv2
import json
import mss
import numpy as np
import os
import pyautogui
import pyautogui
import pygetwindow as gw
import pytesseract
import random
import re
import time
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ------------------------------------------------------------------
# 1. Constants & Waypoints
# ------------------------------------------------------------------

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

TRACK_WIDTH = 37.0          # Rough width of the track (used only for visualisation)
TIME_TO_BEAT = 23.77        # Target finish time you want to beat
EPS_SPEED_START = 0.01      # Speed threshold that starts the timer

# ------------------------------------------------------------------
# 2. Waypoint tolerance & crash detection
# ------------------------------------------------------------------

WAYPOINT_TOLERANCE = 50.0    # metres: car only needs to be within this radius of a waypoint
CRASH_WINDOW = 50           # frames (~0.5 s at 100 Hz) below threshold before penalty is applied
crash_counter = 0          # global counter that counts how long we have been “stuck”

# ------------------------------------------------------------------
# 3. Discretisation for tabular Q-learning
# ------------------------------------------------------------------

SPEED_BINS      = np.linspace(0, 50, 6)          # 5 bins: [0-10),[10-20)…[40-50]
YAW_BINS        = np.linspace(-180, 180, 7)     # every 60°
DIST_BINS       = np.linspace(0, 200, 5)         # 4 distance buckets
CHECKPOINT_BINS = [0, 1, 2]                       # checkpoints (start → mid → finish)

NUM_ACTIONS = 3          # 0: steer left, 1: steer right, 2: no steering
OBS_SHAPE   = (
    len(SPEED_BINS) - 1,
    len(YAW_BINS)   - 1,
    len(DIST_BINS)  - 1,
    len(CHECKPOINT_BINS),
)

# ------------------------------------------------------------------
# 4. Helper utilities
# ------------------------------------------------------------------

def discretise(value, bins):
    """Return the index of the bin that contains `value`."""
    return int(np.digitize([value], bins)[0] - 1)

def press_enter(times: int = 3, delay: float = 0.2):
    """
    Simulate pressing Enter `times` times with a small pause in between.
    The race game usually needs three Enter presses to fully reset.
    """
    for _ in range(times):
        pyautogui.press('enter')
        time.sleep(delay)

# ------------------------------------------------------------------
# 5. HUD Reader – stub (replace!)
# ------------------------------------------------------------------

def read_hud() -> dict:
    """
    Return a dictionary containing the current HUD values.
    Replace this stub with your own method of reading the game's data
    (screen capture + OCR, memory read, API call, etc.).
    """
    # Example stub – randomised for demonstration
    return {
        "elapsed_time": np.random.uniform(0, 30),
        "yaw":          np.random.uniform(-180, 180),
        "pitch":        np.random.uniform(-10, 10),   # unused in this example
        "roll":         np.random.uniform(-5, 5),     # unused
        "speed":        np.random.uniform(0, 50),
        "steer_dir":    np.random.choice([-1, 0, 1]), # left/right/no-steer
        "turn_rate":    np.random.uniform(-200, 200),# unused
        "checkpoint":   np.random.randint(0, 4),
        "total_checkpoint": 3,
        # If you can read the car’s position from the game:
        "car_pos": (np.random.uniform(0, 1000),
                    np.random.uniform(0, 100),
                    np.random.uniform(0, 800)),
    }

# ------------------------------------------------------------------
# 6. Waypoint Navigator
# ------------------------------------------------------------------

class WaypointNavigator:
    """Keeps track of which waypoint we should be heading toward."""
    def __init__(self, waypoints: List[Tuple[float,float,float]]):
        self.waypoints = np.array(waypoints)
        self.total     = len(self.waypoints)

    def current_target_index(self, checkpoint: int) -> int:
        """Return the index of the waypoint we are currently aiming for."""
        return min(checkpoint + 1, self.total - 1)

    def distance_to_target(self,
                           car_pos: Tuple[float,float,float],
                           checkpoint: int) -> float:
        """Distance from the car to the current target, minus tolerance."""
        target_idx = self.current_target_index(checkpoint)
        target     = self.waypoints[target_idx]
        raw_dist   = np.linalg.norm(np.array(car_pos) - target)

        # Subtract tolerance so that the distance is zero once we are close enough
        return max(0.0, raw_dist - WAYPOINT_TOLERANCE)

    def vector_to_target(self,
                         car_pos: Tuple[float,float,float],
                         checkpoint: int) -> np.ndarray:
        """Vector pointing from the car to the target waypoint."""
        target_idx = self.current_target_index(checkpoint)
        target     = self.waypoints[target_idx]
        return target - np.array(car_pos)

# ------------------------------------------------------------------
# 7. State extractor
# ------------------------------------------------------------------

class StateExtractor:
    """
    Converts raw HUD data into a discretised state tuple that the Q-table can index.
    """
    def __init__(self, navigator: WaypointNavigator):
        self.nav = navigator

    def extract(self, hud: dict) -> Tuple[int,int,int,int]:
        s_speed = discretise(hud["speed"], SPEED_BINS)
        s_yaw   = discretise(hud["yaw"], YAW_BINS)

        dist    = self.nav.distance_to_target(
                    hud.get("car_pos", (0,0,0)), hud["checkpoint"])
        s_dist  = discretise(dist, DIST_BINS)

        s_ckpt  = hud["checkpoint"]   # already 0-2

        return (s_speed, s_yaw, s_dist, s_ckpt)

# ------------------------------------------------------------------
# 8. Q-learning agent
# ------------------------------------------------------------------

class QAgent:
    """
    Tabular Q-learner with ε-greedy exploration and an optional exploration bonus.
    """
    def __init__(self,
                 state_shape: Tuple[int,...],
                 n_actions: int,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 decay_steps=100000):
        self.state_shape = state_shape
        self.n_actions   = n_actions
        self.q_table     = {}          # key: state tuple, value: np.array of Qs
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.step_cnt    = 0

    def _q(self, state):
        """Return the Q-values for a given state (initialise if unseen)."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def act(self, state) -> int:
        """
        ε-greedy action selection.
        Exploration bonus: +0.01 for unseen actions to encourage trying them.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        q_vals = self._q(state)
        bonus   = 0.01 * (self.q_table[state] == 0).astype(float)
        q_bon   = q_vals + bonus
        return int(np.argmax(q_bon))

    def update(self,
               state: Tuple[int,...],
               action: int,
               reward: float,
               next_state: Tuple[int,...],
               done: bool):
        """Standard Q-learning TD update."""
        q_current = self._q(state)[action]
        q_next_max = np.max(self._q(next_state))
        target = reward + (0 if done else self.gamma * q_next_max)
        td_error = target - q_current
        self.q_table[state][action] += self.alpha * td_error

        # ε-decay schedule
        self.step_cnt += 1
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start -
                           (self.epsilon_start-self.epsilon_end) *
                           self.step_cnt / self.decay_steps)

# ------------------------------------------------------------------
# 9. Reward function
# ------------------------------------------------------------------

def compute_reward(hud: dict, done: bool) -> float:
    """
    The reward is a weighted sum of:
      * negative elapsed time (faster = higher)
      * checkpoint progress (bonus for each new checkpoint)
      * crash penalty applied only after being stuck for CRASH_WINDOW frames
      * finish bonus when the race ends successfully.
    """
    global crash_counter

    r = -hud["elapsed_time"] / TIME_TO_BEAT
    r += 0.1 * hud["checkpoint"]

    # Crash penalty logic
    if hud["speed"] < EPS_SPEED_START:
        crash_counter += 1
        if crash_counter > CRASH_WINDOW:          # only penalise after we have been stuck long enough
            r -= 10.0                             # heavy negative reward
    else:
        crash_counter = 0

    if done and hud["checkpoint"] == hud["total_checkpoint"]:
        r += 50.0

    return r

# ------------------------------------------------------------------
# 10. Action executor (arrow keys)
# ------------------------------------------------------------------

ACTION_KEYS = {
    0: "left",   # steer left
    1: "right",  # steer right
    2: None,     # no steering (straight)
}

def send_action(action: int, gas: bool, brake: bool):
    """
    Convert an action index into actual keyboard events.
    The game usually expects:
      * Arrow-Left / Arrow-Right for steering
      * Arrow-Up   for acceleration
      * Arrow-Down for braking (optional)
    """
    steer_key = ACTION_KEYS[action]
    if steer_key:
        pyautogui.keyDown(steer_key)

    if gas:
        pyautogui.keyDown("up")
    if brake:
        pyautogui.keyDown("down")

    # Let the game process the key press for a few milliseconds
    time.sleep(0.01)

    # Release all keys we pressed
    if steer_key:
        pyautogui.keyUp(steer_key)
    if gas:
        pyautogui.keyUp("up")
    if brake:
        pyautogui.keyUp("down")

# ------------------------------------------------------------------
# 11. Episode runner – with first-4-s gas rule
# ------------------------------------------------------------------

def run_episode(agent: QAgent,
                navigator: WaypointNavigator,
                extractor: StateExtractor,
                max_steps=5000):
    """
    One full race episode.
    The only change from the previous version is the “first 4 s of constant acceleration” rule.
    """
    # --- reset game --------------------------------------------------
    press_enter()
    time.sleep(1.0)          # give the game a moment to load

    done = False
    step = 0
    cumulative_reward = 0.0

    while not done and step < max_steps:
        hud   = read_hud()
        state = extractor.extract(hud)

        action = agent.act(state)

        # ---------- NEW: force gas for the first 4 s ----------
        if hud["elapsed_time"] <= 4.0:
            gas = True
        else:
            gas = hud["speed"] < 45          # original rule

        brake = False

        send_action(action, gas, brake)

        # read new HUD after the action has taken effect
        hud_next   = read_hud()
        next_state = extractor.extract(hud_next)

        reward    = compute_reward(hud_next,
                                   done=(hud_next["checkpoint"] == hud_next["total_checkpoint"]))
        cumulative_reward += reward

        agent.update(state, action, reward, next_state,
                     done=(hud_next["checkpoint"] == hud_next["total_checkpoint"]))

        step += 1

    # Episode finished – restart race automatically
    press_enter(times=3)

    return cumulative_reward, hud_next["elapsed_time"]

# ------------------------------------------------------------------
# 12. Main training loop
# ------------------------------------------------------------------

def main():
    navigator   = WaypointNavigator(WAYPOINTS)
    extractor   = StateExtractor(navigator)
    agent       = QAgent(state_shape=OBS_SHAPE,
                         n_actions=NUM_ACTIONS)

    best_time   = float("inf")
    episodes    = 200

    for ep in range(1, episodes + 1):
        cum_reward, elapsed = run_episode(agent, navigator, extractor)
        print(f"Episode {ep:3d} | reward={cum_reward:.2f} | time={elapsed:.2f}s")

        # Keep track of the best finish time
        if elapsed < best_time:
            best_time = elapsed
            print(f"*** NEW BEST TIME: {best_time:.2f}s ***")
            # (Optional) you could capture the action sequence here

    # Persist the Q-table for later use or fine-tuning
    import pickle
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)

if __name__ == "__main__":
    main()

