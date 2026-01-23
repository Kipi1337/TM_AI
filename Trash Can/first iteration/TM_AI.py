import random
import csv

# ================================
# Asetukset
# ================================
kartta = "A01-Race"
output_script = "ai_run.script"
total_frames = 6000        # esimerkki, ~60s
frame_time = 0.01         # 100 FPS
STEER_MIN = -3000
STEER_MAX = 3000

# ================================
# Funktio rewardin laskuun (simulaatio)
# ================================
def compute_reward(state, done):
    reward = state.get("distance_progress", 0)
    if done:
        reward += 1000  # iso bonus maaliin
    if state.get("offtrack", False):
        reward -= 10
    return reward

# ================================
# Luodaan TMInterface-skriptin inputit
# ================================
frame_inputs = []

for i in range(total_frames):
    # AI ennustaa inputit tässä, toistaiseksi random
    acc = True  # kaasu aina
    brake = random.random() < 0.05  # satunnaisesti jarru
    steer = random.randint(STEER_MIN, STEER_MAX)
    
    frame_inputs.append({"frame": i, "acc": acc, "brake": brake, "steer": steer})

# ================================
# Kirjoitetaan TMInterface-skripti
# ================================
with open(output_script, "w") as f:
    f.write(f"# AI run script for {kartta}\n")
    f.write("0.00 press up\n")  # Kaasu heti alussa

    for frame in frame_inputs:
        start_time = round(frame["frame"] * frame_time, 2)
        end_time = round((frame["frame"] + 1) * frame_time, 2)

        # Steer
        f.write(f"{start_time}-{end_time} steer {frame['steer']}\n")
        # Brake
        if frame["brake"]:
            f.write(f"{start_time}-{end_time} press down\n")
            f.write(f"{end_time} rel down\n")

    # Lopeta kaasu
    f.write(f"{round(total_frames*frame_time,2)} rel up\n")

print(f"TMInterface AI-run script generated: {output_script}")
print(f"Run in console: load(\"{output_script}\")")
