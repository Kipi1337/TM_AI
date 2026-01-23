import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
CSV_FILE = "tm_ai_training_ready.csv"
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

# -----------------------------
# Map human actions to indices
# -----------------------------
action_map = {"up":0, "down":1, "left":2, "right":3, "gas":4, "none":5}
num_actions = len(action_map)

def action_to_index(action):
    return action_map.get(action, action_map["none"])

# -----------------------------
# Dataset Class
# -----------------------------
class TrackDataset(Dataset):
    def __init__(self, csv_file):
        self.states = []
        self.actions = []

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
                speed = float(row["speed"])
                turning = float(row["turning"])
                dist_cp = float(row["dist_to_next_cp"])
                state = [x, y, z, speed, turning, dist_cp]
                self.states.append(state)
                self.actions.append(action_to_index(row["action"]))

        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# -----------------------------
# Simple Neural Network
# -----------------------------
class TrackNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(TrackNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x

# -----------------------------
# Load Data
# -----------------------------
dataset = TrackDataset(CSV_FILE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Initialize Model
# -----------------------------
model = TrackNet(input_size=6, output_size=num_actions)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    for states, actions in dataloader:
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "tracknet_model.pth")
print("Training complete. Model saved as tracknet_model.pth")
