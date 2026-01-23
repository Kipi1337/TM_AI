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
    {"X": (967, 977), "Y": (19, 29), "Z": (300, 310)},
    {"X": (492, 502), "Y": (78, 88), "Z": (171, 181)},
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
# Exploration Reward
# -----------------------------
grid_size = 20
visited_cells = set()
estimated_total_cells = 1000

def exploration_reward(x, z):
    cell = (int(x/grid_size), int(z/grid_size))
    if cell not in visited_cells:
        visited_cells.add(cell)
        return 2
    return 0

# -----------------------------
# Persistent Learned Actions (Weighted + Punishment)
# -----------------------------
learned_actions_file = "tm_ai_learned_actions.pkl"
if os.path.exists(learned_actions_file):
    with open(learned_actions_file, "rb") as f:
        learned_actions = pickle.load(f)
    print("Loaded learned actions from disk.")
else:
    learned_actions = {}

punished_actions_file = "tm_ai_punished_actions.pkl"
if os.path.exists(punished_actions_file):
    with open(punished_actions_file, "rb") as f:
        punished_actions = pickle.load(f)
else:
    punished_actions = {}

# -----------------------------
# Update Positive Memory
# -----------------------------
def update_learned_actions(cp_index, path_positions, path_actions, reward_history):
    cp_name = f"cp{cp_index+1}"
    if cp_name not in learned_actions:
        learned_actions[cp_name] = []

    for pos, action, reward in zip(path_positions, path_actions, reward_history):
        if reward < 0:
            continue
        found = False
        for entry in learned_actions[cp_name]:
            px, py, pz = entry["position"]
            dist = ((pos[0]-px)**2 + (pos[1]-py)**2 + (pos[2]-pz)**2)**0.5
            if dist < 5:
                entry["action_counts"][action] = entry["action_counts"].get(action, 0) + 1
                found = True
                break
        if not found:
            learned_actions[cp_name].append({
                "position": pos,
                "action_counts": {action: 1}
            })

    with open(learned_actions_file, "wb") as f:
        pickle.dump(learned_actions, f)

def recall_action(x, y, z, cp_index):
    cp_name = f"cp{cp_index+1}"
    if cp_name not in learned_actions:
        return None
    closest = None
    min_dist = float("inf")
    for entry in learned_actions[cp_name]:
        px, py, pz = entry["position"]
        dist = ((x - px)**2 + (y - py)**2 + (z - pz)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            closest = entry
    if closest and min_dist < 5:
        return max(closest["action_counts"], key=lambda a: closest["action_counts"][a])
    return None

# -----------------------------
# Punishment Memory
# -----------------------------
def update_punished_actions(path_positions, path_actions):
    for pos, action in zip(path_positions, path_actions):
        key = (round(pos[0],1), round(pos[1],1), round(pos[2],1))
        if key not in punished_actions:
            punished_actions[key] = {}
        punished_actions[key][action] = punished_actions[key].get(action, 0) + 1
    with open(punished_actions_file, "wb") as f:
        pickle.dump(punished_actions, f)

def avoid_punished_actions(x, y, z, action_idx):
    key = (round(x,1), round(y,1), round(z,1))
    if key in punished_actions:
        action = action_list[action_idx]
        if action in punished_actions[key]:
            safe_actions = [a for a in action_list if a != action]
            if safe_actions:
                return action_list.index(random.choice(safe_actions))
    return action_idx

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
state_size = 6
action_list = ["up", "down", "left", "right", "none"]
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
path_positions = []
reward_path = []
reward_history = deque(maxlen=200)

# -----------------------------
# Main Loop
# -----------------------------
try:
    while True:
        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)
        speed = pm.read_int(SPEED_ADDR)
        turning = pm.read_float(TURN_ADDR)
        racetime = pm.read_int(RACETIME_ADDR)

        curr_pos = (x, y, z)
        path_positions.append(curr_pos)

        # Punishment check with retry
        if y < 15:
            print("Fell off track! Recording punished actions and retrying...")
            update_punished_actions(path_positions, path_actions)
            path_actions = []
            path_positions = []
            reward_path = []
            prev_dist_to_cp = 0
            next_cp_index = 0

            # Press Enter to retry
            keyboard.press(Key.enter)
            time.sleep(0.2)
            keyboard.release(Key.enter)
            time.sleep(2)  # wait for game reset
            continue

        next_cp = checkpoints[next_cp_index] if next_cp_index < len(checkpoints) else finish_line
        checkpoint_reached = in_zone(x, y, z, next_cp)
        finished = in_zone(x, y, z, finish_line)

        dist_to_cp = distance_to_zone(x, y, z, next_cp)
        state = [x, y, z, speed, turning, dist_to_cp]

        # Reward
        reward = compute_reward(prev_dist_to_cp, dist_to_cp, speed, checkpoint_reached, finished)
        reward += exploration_reward(x, z)
        prev_dist_to_cp = dist_to_cp
        reward_path.append(reward)
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
        recalled = recall_action(x, y, z, next_cp_index)
        if recalled:
            action_idx = action_list.index(recalled)
        action_idx = avoid_punished_actions(x, y, z, action_idx)
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

        # Checkpoint reached
        if checkpoint_reached:
            update_learned_actions(next_cp_index, path_positions, path_actions, reward_path)
            path_actions = []
            path_positions = []
            reward_path = []
            next_cp_index += 1

        # Finish line
        if finished:
            prev_dist_to_cp = 0
            next_cp_index = 0
            path_actions = []
            path_positions = []
            reward_path = []

        time.sleep(dt)

except KeyboardInterrupt:
    print("RL loop stopped by user")
    for key_name, pressed in keys_state.items():
        k = key_map.get(key_name)
        if pressed and k:
            keyboard.release(k)
