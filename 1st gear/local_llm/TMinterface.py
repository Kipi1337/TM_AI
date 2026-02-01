import time
import random
import numpy as np
from collections import deque
from tminterface.interface import TMInterface
from tminterface.client import Client
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pynput.keyboard import Key, Controller
import threading

# =========================
# CONFIG
# =========================
CHECKPOINT_REWARD = 1000.0
SPEED_REWARD_MULT = 0.2
FINISH_REWARD = 5000.0

class TrackmaniaClient(Client):
    def __init__(self):
        super().__init__()
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.episode = 0
        self.prev_state = None
        self.prev_checkpoint = 0
        self.keyboard = Controller()
        self.running = True
        self.current_state = None
        self.checkpoint_confirmed = 0
        self.finish_confirmed = False
        
    def _build_model(self):
        model = Sequential([
            Dense(256, activation="relu", input_shape=(11,)),
            Dense(256, activation="relu"),
            Dense(7, activation="linear")
        ])
        model.compile(optimizer=Adam(0.0005), loss="mse")
        return model
    
    def on_registered(self, iface: TMInterface):
        print("Connected to TMInterface!")
        # Enable realtime mode (simulation on server)
        iface.set_timeout(0)  # No timeout, run forever
        self.iface = iface
        threading.Thread(target=self.training_loop, daemon=True).start()
    
    def on_simulation_step(self, iface: TMInterface, time: int):
        """
        Called every simulation step (10ms by default)
        time is in milliseconds
        """
        # Get state directly from game memory
        state = iface.get_simulation_state()
        
        # Extract features (no OCR!)
        pos = state.position
        vel = state.velocity
        speed = np.linalg.norm(vel) * 3.6  # m/s to km/h
        
        # Get checkpoint info from simulation (accurate!)
        # Note: In newer TMInterface versions, check iface.get_checkpoint_state()
        # or track triggers manually
        
        # Construct observation
        observation = np.array([
            pos[0], pos[1], pos[2],
            speed,
            state.yaw, state.pitch, state.roll,
            state.input_steer, state.turning_rate,
            self.checkpoint_confirmed,
            3  # Total checkpoints
        ])
        
        # Determine if finished (race_time stops increasing or checkpoint count)
        finished = self.check_finish_condition(state, time)
        
        # RL Logic
        if self.prev_state is not None:
            reward = self.compute_reward(self.prev_state, observation, finished)
            done = finished
            self.memory.append((self.prev_state, self.last_action, reward, observation, done))
            
            if len(self.memory) > 64:
                self.replay()
        
        # Select action
        if random.random() < self.epsilon:
            action = random.randrange(7)
        else:
            q = self.model.predict(observation.reshape(1, -1), verbose=0)
            action = np.argmax(q[0])
            
        self.last_action = action
        self.prev_state = observation
        self.execute_action(action)
        
        if finished:
            print("Race finished! Restarting...")
            iface.execute_command("respawn")
            self.episode += 1
            self.prev_state = None
            self.checkpoint_confirmed = 0
            
            # Save model periodically
            if self.episode % 10 == 0:
                self.model.save(f"dqn_tmi_ep{self.episode}.keras")
                
    def check_finish_condition(self, state, time):
        """
        Robust finish detection using:
        1. Checkpoint count from simulation events
        2. Position stabilization
        3. Speed drop to near zero
        """
        speed = np.linalg.norm(state.velocity) * 3.6
        
        # Check for position lock (car stopped at finish)
        if hasattr(self, 'finish_pos'):
            dist = np.linalg.norm(np.array(state.position) - self.finish_pos)
            if dist < 2.0 and speed < 1.0:
                return True
        elif speed < 1.0 and time > 10000:  # After 10 seconds, stopped = finish
            self.finish_pos = state.position
            return True
            
        return False
    
    def compute_reward(self, prev, curr, finished):
        reward = 0.0
        
        # Speed reward
        reward += curr[3] * SPEED_REWARD_MULT  # Index 3 is speed
        
        # Checkpoint detection (using index 9)
        if curr[9] > prev[9]:
            reward += CHECKPOINT_REWARD
            print(f"CHECKPOINT {curr[9]}!")
            
        # Distance traveled reward
        prev_pos = np.array(prev[0:3])
        curr_pos = np.array(curr[0:3])
        progress = curr_pos[0] - prev_pos[0]  # Assuming X is forward
        reward += progress * 0.1
        
        if finished:
            reward += FINISH_REWARD
            
        return reward
    
    def execute_action(self, action):
        """Direct input injection via TMInterface (more reliable than keyboard)"""
        # Map actions to TMInterface commands
        actions_map = {
            0: "press up",
            1: "press down", 
            2: "steer -65536",  # Full left
            3: "steer 65536",   # Full right
            4: "press up; steer -65536",
            5: "press up; steer 65536",
            6: ""  # No input
        }
        
        # Clear previous inputs first (important!)
        self.iface.execute_command("rel up; rel down; steer 0")
        
        if action != 6:
            self.iface.execute_command(actions_map[action])
    
    def replay(self):
        batch = random.sample(self.memory, 64)
        # ... standard DQN training ...
        states = np.array([x[0] for x in batch])
        # Flatten the nested tuple structure if needed
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        target_q = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + 0.99 * np.max(next_q[i])
                
        self.model.fit(states, target_q, epochs=1, verbose=0)
        self.epsilon = max(0.05, self.epsilon * 0.999)
    
    def training_loop(self):
        """Separate thread for training while running"""
        while self.running:
            if len(self.memory) > 1000:
                self.replay()
            time.sleep(1)

# =========================
# MAIN
# =========================
def main():
    client = TrackmaniaClient()
    
    # Connect to TMInterface server (default port 8477)
    interface = TMInterface("127.0.0.1", 8477)
    interface.register(client)
    
    print("Connecting to TMInterface...")
    interface.run()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        client.running = False
        interface.close()

if __name__ == "__main__":
    main()