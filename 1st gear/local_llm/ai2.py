import os
import time
import random
import pickle
import threading
import numpy as np
from collections import deque, namedtuple
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from pynput.keyboard import Key, Controller
import pygetwindow as gw
import mss
from PIL import Image
import cv2
import pytesseract
import re

# =========================
# ARCHITECTURE: STATE MACHINE
# =========================
class RacePhase(Enum):
    COUNTDOWN = auto()
    RACING = auto()
    FINISHED = auto()
    CRASHED = auto()
    STUCK = auto()

@dataclass
class RaceContext:
    """Immutable context container"""
    phase: RacePhase = RacePhase.COUNTDOWN
    checkpoint: int = 0
    finish_position: Optional[np.ndarray] = None
    start_time: float = field(default_factory=time.time)
    max_speed: float = 0.0
    distance_traveled: float = 0.0

# =========================
# SAFETY SYSTEM
# =========================
class SafetyMonitor:
    """Emergency recovery and monitoring"""
    def __init__(self):
        self.emergency_stop = threading.Event()
        self.last_heartbeat = time.time()
        self.lock = threading.Lock()
        
    def heartbeat(self):
        self.last_heartbeat = time.time()
        
    def is_stuck(self):
        return (time.time() - self.last_heartbeat) > 10.0
        
    def emergency_reset(self):
        print("🚨 EMERGENCY RESET")
        keyboard = Controller()
        for key in [Key.up, Key.down, Key.left, Key.right]:
            keyboard.release(key)
        keyboard.press(Key.enter)
        time.sleep(0.5)
        keyboard.release(Key.enter)

# =========================
# SENSOR FUSION (Multiple data sources)
# =========================
class SensorFusion:
    """Combines OCR with temporal consistency and physics prediction"""
    def __init__(self):
        self.history = deque(maxlen=5)
        self.predictor = PhysicsPredictor()  # Dead reckoning when OCR fails
        
    def update(self, raw_hud: dict, valid: bool) -> dict:
        if valid:
            self.history.append(raw_hud)
            self.predictor.update(raw_hud)
            return self._temporal_filter()
        else:
            # Dead reckoning when OCR fails
            return self.predictor.predict()
    
    def _temporal_filter(self):
        """Median filter for stability"""
        if len(self.history) < 3:
            return self.history[-1]
            
        fields = ['speed', 'checkpoint_current', 'position', 'turning_rate']
        result = {}
        
        for field in fields:
            values = [h[field] for h in self.history]
            if field == 'position':
                result[field] = tuple(np.median([v[i] for v in values]) for i in range(3))
            else:
                result[field] = float(np.median(values))
                
        return result

class PhysicsPredictor:
    """Simple physics model for dead reckoning"""
    def __init__(self):
        self.last_state = None
        self.velocity = np.zeros(3)
        self.last_time = time.time()
        
    def update(self, state):
        if self.last_state:
            dt = time.time() - self.last_time
            pos = np.array(state['position'])
            last_pos = np.array(self.last_state['position'])
            self.velocity = (pos - last_pos) / (dt + 0.001)
            
        self.last_state = state.copy()
        self.last_time = time.time()
        
    def predict(self, dt=0.1):
        """Predict next state based on physics"""
        if self.last_state is None:
            return None
            
        predicted = self.last_state.copy()
        new_pos = np.array(predicted['position']) + self.velocity * dt
        predicted['position'] = tuple(new_pos)
        predicted['speed'] = np.linalg.norm(self.velocity) * 3.6  # m/s to km/h
        return predicted

# =========================
# ADVANCED REPLAY BUFFER (PER)
# =========================
class PrioritizedExperienceReplay:
    def __init__(self, capacity=200000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, experience, priority=1.0):
        max_prio = self.priorities.max() if self.memory else 1.0
        priority = max(priority, max_prio)
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=128):
        if len(self.memory) == 0:
            return [], [], []
            
        probs = self.priorities[:len(self.memory)]
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = (prio + 1e-6) ** self.alpha

# =========================
# Dueling DQN Architecture
# =========================
class DuelingDQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.dense1 = Dense(512, activation='relu')
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(256, activation='relu')
        
        # Value stream
        self.value_dense = Dense(128, activation='relu')
        self.value_out = Dense(1)
        
        # Advantage stream
        self.adv_dense = Dense(128, activation='relu')
        self.adv_out = Dense(action_size)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        
        value = self.value_dense(x)
        value = self.value_out(value)
        
        advantage = self.adv_dense(x)
        advantage = self.adv_out(advantage)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values

# =========================
# PHASE-AWARE AGENT
# =========================
class PhaseAwareAgent:
    def __init__(self):
        self.model = DuelingDQN(11, 7)
        self.model.compile(optimizer=Adam(0.0003))
        self.target_model = DuelingDQN(11, 7)
        self.memory = PrioritizedExperienceReplay()
        self.epsilon = 1.0
        self.episode_count = 0
        
        # Phase-specific strategies
        self.phase_strategies = {
            RacePhase.COUNTDOWN: self._countdown_strategy,
            RacePhase.RACING: self._racing_strategy,
            RacePhase.FINISHED: self._finished_strategy,
            RacePhase.STUCK: self._stuck_strategy
        }
        
    def _countdown_strategy(self, state):
        # Always accelerate at start
        return 0  # Key.up
    
    def _racing_strategy(self, state):
        # Normal epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(7)
        q = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q[0])
    
    def _finished_strategy(self, state):
        return 6  # No input
        
    def _stuck_strategy(self, state):
        # Random recovery maneuvers
        return random.choice([1, 2, 3, 6])  # Brake or turn
    
    def act(self, state, phase):
        return self.phase_strategies[phase](state)
    
    def train(self):
        if len(self.memory.memory) < 128:
            return
            
        batch, indices, weights = self.memory.sample()
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Double DQN update
        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        q_next_target = self.target_model.predict(next_states, verbose=0)
        
        targets = q_current.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # Double DQN: use model to select action, target to evaluate
                best_action = np.argmax(q_next[i])
                targets[i, actions[i]] = rewards[i] + 0.99 * q_next_target[i, best_action]
        
        # Weighted loss
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            selected_q = tf.reduce_sum(q_values * tf.one_hot(actions, 7), axis=1)
            loss = tf.reduce_mean(weights * (targets[np.arange(len(batch)), actions] - selected_q) ** 2)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Update priorities
        td_errors = np.abs(targets[np.arange(len(batch)), actions] - 
                          self.model.predict(states, verbose=0)[np.arange(len(batch)), actions])
        self.memory.update_priorities(indices, td_errors)

# =========================
# ORCHESTRATOR
# =========================
class TrackmaniaOrchestrator:
    def __init__(self):
        self.context = RaceContext()
        self.agent = PhaseAwareAgent()
        self.sensors = SensorFusion()
        self.safety = SafetyMonitor()
        self.controller = SmoothController()
        
    def run_episode(self):
        self.context = RaceContext()
        self.context.phase = RacePhase.COUNTDOWN
        
        while self.context.phase not in [RacePhase.FINISHED, RacePhase.CRASHED]:
            # Safety check
            if self.safety.is_stuck():
                self.safety.emergency_reset()
                self.context.phase = RacePhase.CRASHED
                break
            
            # Sensing
            raw_hud, valid = self.read_sensors()
            hud = self.sensors.update(raw_hud, valid)
            
            # Phase detection
            new_phase = self.detect_phase(hud)
            if new_phase != self.context.phase:
                print(f"Phase transition: {self.context.phase} -> {new_phase}")
                self.context.phase = new_phase
            
            # Action
            state = self.hud_to_state(hud)
            action = self.agent.act(state, self.context.phase)
            oscillating = self.controller.execute(action)
            
            # Environment step
            time.sleep(0.1)
            next_hud, next_valid = self.read_sensors()
            next_state = self.hud_to_state(self.sensors.update(next_hud, next_valid))
            
            # Reward shaping based on phase
            reward = self.calculate_phase_reward(hud, next_hud, oscillating)
            
            # Store
            done = self.context.phase in [RacePhase.FINISHED, RacePhase.CRASHED]
            self.agent.memory.add(
                Experience(state, action, reward, next_state, done),
                priority=abs(reward) + 0.01
            )
            
            self.safety.heartbeat()
            
        # Post-episode
        for _ in range(50):
            self.agent.train()
        self.agent.episode_count += 1
        
        if self.agent.episode_count % 10 == 0:
            self.agent.target_model.set_weights(self.agent.model.get_weights())
            
    def detect_phase(self, hud):
        if hud["checkpoint_current"] >= 3 and hud["speed"] < 0.5:
            return RacePhase.FINISHED
        elif hud["speed"] < 1.0 and self.context.phase == RacePhase.RACING:
            return RacePhase.STUCK
        elif self.context.phase == RacePhase.COUNTDOWN and hud["speed"] > 1.0:
            return RacePhase.RACING
        return self.context.phase
        
    def calculate_phase_reward(self, hud, next_hud, oscillating):
        base = 0.0
        
        if self.context.phase == RacePhase.RACING:
            base += next_hud["speed"] * 0.2
            base += (next_hud["checkpoint_current"] - hud["checkpoint_current"]) * 1000
            
        if oscillating:
            base -= 20
            
        return base
        
    def read_sensors(self):
        # Your OCR code here, returns (hud_dict, is_valid)
        pass
        
    def hud_to_state(self, hud):
        return np.array([
            hud["position"][0], hud["position"][1], hud["position"][2],
            hud["speed"], hud["yaw"], hud["pitch"], hud["roll"],
            hud["steer_direction"], hud["turning_rate"],
            hud["checkpoint_current"], 0  # Normalized
        ])

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])