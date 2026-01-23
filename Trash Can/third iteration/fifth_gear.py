import pymem
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pynput.keyboard import Key, Controller
from collections import deque
import pickle
import os

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
# Reward Function
# -----------------------------
def compute_reward(prev_dist_to_cp, curr_dist_to_cp, speed, checkpoint_reached=False, finished=False):
    reward = 0
    distance_delta = prev_dist_to_cp - curr_dist_to_cp
    bonus_units = 50
    reward += int(distance_delta / bonus_units) * 5

    if checkpoint_reached:
        reward += 100
    if finished:
        reward += 1000

    if speed > 50:
        reward += 1

    return reward

# -----------------------------
# Exploration reward
# -----------------------------
grid_size = 20
visited_cells = set()
estimated_total_cells = 1000  # rough estimate

def exploration_reward(x, z):
    cell = (int(x/grid_size), int(z/grid_size))
    if cell not in visited_cells:
        visited_cells.add(cell)
        return 2
    return 0

# -----------------------------
# Persistent Learned Actions
# -----------------------------
learned_actions_file = "tm_ai_learned_actions.pkl"
if os.path.exists(learned_actions_file):
    with open(learned_actions_file, "rb") as f:
        learned_actions = pickle.load(f)
    print("Loaded learned actions from disk.")
else:
    learned_actions = {}

def recall_action(x, y, z):
    for cp_name, data in learned_actions.items():
        zone = {"X": (data["position"][0]-5, data["position"][0]+5),
                "Y": (data["position"][1]-5, data["position"][1]+5),
                "Z": (data["position"][2]-5, data["position"][2]+5)}
        if in_zone(x, y, z, zone):
            return random.choice(data["preferred_actions"])
    return None

def update_learned_actions(cp_index, path_actions, x, y, z):
    cp_name = f"cp{cp_index+1}"
    learned_actions[cp_name] = {"position": (x, y, z), "preferred_actions": path_actions}
    with open(learned_actions_file, "wb") as f:
        pickle.dump(learned_actions, f)

# -----------------------------
# DQN Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Hyperparameters
# -----------------------------
state_size = 6  # x, y, z, speed, turning, dist_to_cp
action_list = ["up", "down", "left", "right", "gas", "none"]
action_size = len(action_list)
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3
batch_size = 32
memory_size = 5000

# -----------------------------
# DQN Agent
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
criterion = nn.MSELoss()
memory = deque(maxlen=memory_size)

def select_action(state, reverse_active):
    if reverse_active:
        return action_list.index("none")
    # Check memory recall
    x, y, z = state[0], state[1], state[2]
    recalled = recall_action(x, y, z)
    if recalled:
        return action_list.index(recalled)
    # Otherwise DQN epsilon-greedy
    if random.random() < epsilon:
        return random.randint(0, action_size-1)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return torch.argmax(q_values).item()

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def train_dqn():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + gamma * next_q * (1 - dones)
    loss = criterion(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------
# TrackMania Interface
# -----------------------------
pm = pymem.Pymem("TmForever.exe")
keyboard = Controller()
keys_state = {a: False for a in action_list}
key_map = {
    "up": Key.up,
    "down": Key.down,
    "left": Key.left,
    "right": Key.right,
    "gas": Key.up,
    "none": None
}

# -----------------------------
# Stuck Recovery
# -----------------------------
slow_speed_threshold = 5
slow_time_limit = 10
reverse_duration = 3
slow_timer = 0
reverse_timer = 0
reverse_active = False
reverse_key = Key.down

prev_dist_to_cp = 0
next_cp_index = 0
dt = 0.05
path_actions = []

# -----------------------------
# Reward Feedback Tracking
# -----------------------------
reward_history = deque(maxlen=200)

# -----------------------------
# Main Loop
# -----------------------------
try:
    while True:
        # Telemetry
        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)
        speed = pm.read_int(SPEED_ADDR)
        turning = pm.read_float(TURN_ADDR)
        racetime = pm.read_int(RACETIME_ADDR)

        curr_pos = (x, y, z)
        next_cp = checkpoints[next_cp_index] if next_cp_index < len(checkpoints) else finish_line
        checkpoint_reached = in_zone(x, y, z, next_cp)
        finished = in_zone(x, y, z, finish_line)

        if checkpoint_reached:
            update_learned_actions(next_cp_index, path_actions, x, y, z)
            path_actions = []
            next_cp_index += 1

        dist_to_cp = distance_to_zone(x, y, z, next_cp)
        state = [x, y, z, speed, turning, dist_to_cp]

        # Reward
        reward = compute_reward(prev_dist_to_cp, dist_to_cp, speed, checkpoint_reached, finished)
        reward += exploration_reward(x, z)
        prev_dist_to_cp = dist_to_cp

        # Track reward history for consistency
        reward_history.append(reward)
        avg_reward = sum(reward_history)/len(reward_history)
        exploration_pct = len(visited_cells) / estimated_total_cells

        # Stuck Recovery
        if not reverse_active:
            if speed < slow_speed_threshold:
                slow_timer += dt
            else:
                slow_timer = 0
            if slow_timer >= slow_time_limit:
                reverse_active = True
                reverse_timer = 0
                print("Car stuck! Reversing for 3 seconds...")
                keyboard.press(reverse_key)
                for k_name, pressed in keys_state.items():
                    k = key_map.get(k_name)
                    if pressed and k and k != reverse_key:
                        keyboard.release(k)
                        keys_state[k_name] = False
        else:
            reverse_timer += dt
            if reverse_timer >= reverse_duration:
                keyboard.release(reverse_key)
                reverse_active = False
                slow_timer = 0
                print("Reverse done. Resuming AI actions...")
            else:
                time.sleep(dt)
                continue

        # Action Selection
        action_idx = select_action(state, reverse_active)
        action = action_list[action_idx]
        path_actions.append(action)

        # Release keys not needed
        for key_name, pressed in keys_state.items():
            if key_name != action and pressed:
                k = key_map.get(key_name)
                if k: keyboard.release(k)
                keys_state[key_name] = False

        # Press selected key
        k = key_map.get(action)
        if k and not keys_state[action]:
            keyboard.press(k)
            keys_state[action] = True

        # Feedback
        print(f"Pos: ({x:.2f},{y:.2f},{z:.2f}) | Speed: {speed} | DistCP: {dist_to_cp:.2f} | Action: {action} | Reward: {reward:.2f} | AvgReward: {avg_reward:.2f} | Explore: {exploration_pct*100:.1f}% | Reverse: {reverse_active}")

        # Store & Train
        next_state = [x, y, z, speed, turning, dist_to_cp]
        done = finished
        store_transition(state, action_idx, reward, next_state, done)
        train_dqn()

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Reset after finish
        if finished:
            prev_dist_to_cp = 0
            next_cp_index = 0
            path_actions = []

        time.sleep(dt)

except KeyboardInterrupt:
    print("RL loop stopped by user")
    for key_name, pressed in keys_state.items():
        k = key_map.get(key_name)
        if pressed and k:
            keyboard.release(k)
