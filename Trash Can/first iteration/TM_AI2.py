import random
import csv

# ================================
# Asetukset
# ================================
kartta = "A01-Race"
output_script = "ai_run.script"
telemetry_csv = "telemetry.csv"
total_frames = 6000
frame_time = 0.01
STEER_MIN = -3000
STEER_MAX = 3000

# ================================
# AI funktio (placeholder)
# ================================
# Tässä vaiheessa AI on random. Myöhemmin voit korvata mallilla.
def ai_predict(state):
    """
    state: dictionary, sisältää telemetry per frame
    palauttaa action: {'acc': bool, 'brake': bool, 'steer': int}
    """
    acc = True
    brake = random.random() < 0.05
    steer = random.randint(STEER_MIN, STEER_MAX)
    return {"acc": acc, "brake": brake, "steer": steer}

# ================================
# Luo frame-by-frame AI inputit
# ================================
frame_inputs = []
# Aloitus telemetry
x, y, z = 0.0, 0.0, 0.0
speed = 0.0

for i in range(total_frames):
    # Luo state AI:lle
    state = {
        "x": x, "y": y, "z": z, "speed": speed,
        "frame": i
    }
    action = ai_predict(state)
    frame_inputs.append(action)

    # Päivitä simuloitu telemetry
    speed = 10 if action["acc"] else max(0, speed - 2)
    x += speed * frame_time
    y += action["steer"] / 10000  # yksinkertainen käännös
    collision = random.random() < 0.01
    if collision:
        speed *= 0.5
    offtrack = random.random() < 0.01

    # Lisää telemetry data
    state.update({
        "steer": action["steer"],
        "acc": int(action["acc"]),
        "brake": int(action["brake"]),
        "collision": int(collision),
        "offtrack": int(offtrack)
    })
    frame_inputs[i].update(state)  # yhdistä state ja action

# ================================
# Laske reward per frame
# ================================
for i, frame in enumerate(frame_inputs):
    reward = frame["speed"] * frame_time
    if frame["collision"]:
        reward -= 10
    if frame["offtrack"]:
        reward -= 20
    # Iso bonus maaliin
    if i == total_frames - 1:
        reward += 1000
    frame["reward"] = round(reward, 2)

# ================================
# Tallenna telemetry CSV
# ================================
with open(telemetry_csv, "w", newline="") as csvfile:
    fieldnames = ["frame","x","y","z","speed","steer","acc","brake","collision","offtrack","reward"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in frame_inputs:
        writer.writerow(row)

print(f"Telemetry CSV generated: {telemetry_csv}")

# ================================
# Luo TMInterface-skripti AI:n ennusteista
# ================================
with open(output_script, "w") as f:
    f.write(f"# AI run script for {kartta}\n")
    f.write("0.00 press up\n")
    for i, frame in enumerate(frame_inputs):
        start_time = round(i * frame_time, 2)
        end_time = round((i+1) * frame_time, 2)
        f.write(f"{start_time}-{end_time} steer {frame['steer']}\n")
        if frame["brake"]:
            f.write(f"{start_time}-{end_time} press down\n")
            f.write(f"{end_time} rel down\n")
    f.write(f"{round(total_frames*frame_time,2)} rel up\n")

print(f"TMInterface AI-run script generated: {output_script}")
print(f"Run in TMInterface console: load(\"{output_script}\")")
