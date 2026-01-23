# tmnf_supervised.py
import numpy as np
import cv2
from mss import mss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from pynput.keyboard import Key, Controller

# -----------------------------
# CONFIG
# -----------------------------
WINDOW_TITLE = "TrackMania Modded Forever (2.12.0) [AI]: TMInterface (2.2.1), CoreMod (1.0.10)"
FPS = 60  # frames per second
IMG_SIZE = 84  # CNN input size
KEY_MAP = {'up':0, 'down':1, 'left':2, 'right':3}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Parse .txt keypress file
# -----------------------------
txt_file = "test.txt"  # your input file

time_actions = []
with open(txt_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        times, key = line.split(" press ")
        start_time, end_time = map(float, times.split("-"))
        key_idx = KEY_MAP[key]
        time_actions.append((start_time, end_time, key_idx))

# Generate frame-by-frame labels
total_time = max(end for _, end, _ in time_actions)
num_frames = int(total_time * FPS) + 1
frame_labels = np.full(num_frames, -1)  # -1 = no key

for start, end, key_idx in time_actions:
    start_frame = int(start * FPS)
    end_frame = int(end * FPS)
    frame_labels[start_frame:end_frame+1] = key_idx

print("Total frames:", num_frames)

# Optional: weight frames by finishing time
finish_time = total_time
weights = np.ones(len(frame_labels)) * (1 / finish_time)

# -----------------------------
# Capture TMNF window
# -----------------------------
sct = mss()

def get_window_rect(title):
    import win32gui
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        raise Exception(f"Window '{title}' not found!")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right, bottom

def capture_frame():
    left, top, right, bottom = get_window_rect(WINDOW_TITLE)
    monitor = {"top": top, "left": left, "width": right-left, "height": bottom-top}
    img = np.array(sct.grab(monitor))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized / 255.0

# -----------------------------
# Collect frames (or load pre-recorded)
# -----------------------------
frames_file = "frames.npy"
if os.path.exists(frames_file):
    X = np.load(frames_file)
    print("Loaded pre-recorded frames:", X.shape)
else:
    print("Capturing frames from TMNF. Please start the race...")
    X = []
    for i in range(len(frame_labels)):
        frame = capture_frame()
        X.append(frame)
        time.sleep(1/FPS)  # approximate timing
    X = np.array(X)
    np.save(frames_file, X)
    print("Frames saved:", X.shape)

# -----------------------------
# Prepare dataset
# -----------------------------
valid_idx = frame_labels != -1  # remove frames with no key
X = X[valid_idx]
y = frame_labels[valid_idx]
sample_weights = weights[valid_idx]

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # add channel dim
y_tensor = torch.tensor(y, dtype=torch.long)
weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# CNN Model
# -----------------------------
class TMNFCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.fc1 = nn.Linear(64*9*9, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = TMNFCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(reduction='none')  # so we can weight by finish time

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(10):
    running_loss = 0.0
    for batch_x, batch_y, batch_w in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        batch_w = batch_w.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss = (loss * batch_w).mean()  # weight by finish time
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "tmnf_cnn_model.pth")
print("Model saved as tmnf_cnn_model.pth")

# -----------------------------
# Optional: Test AI playing
# -----------------------------
keyboard = Controller()
current_keys = set()  # keys currently pressed

reverse_key_map = {
    0: Key.up,    # accelerate
    1: Key.down,  # brake
    2: Key.left,  # steer left
    3: Key.right  # steer right
}

def perform_action(action_idx):
    global current_keys
    desired_keys = {reverse_key_map[action_idx]}  # AI wants this key

    # Release keys that are no longer desired
    for key in current_keys - desired_keys:
        keyboard.release(key)

    # Press new keys
    for key in desired_keys - current_keys:
        keyboard.press(key)

    # Update currently held keys
    current_keys = desired_keys

print("AI ready. Press CTRL+C to stop while playing.")
try:
    while True:
        frame = capture_frame()
        frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_idx = model(frame_tensor).argmax().item()
        perform_action(action_idx)
        time.sleep(1/FPS)
except KeyboardInterrupt:
    print("AI stopped.")
