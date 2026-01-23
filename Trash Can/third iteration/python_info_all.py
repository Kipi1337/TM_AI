import pymem
import time
import csv

# Attach to TrackMania
pm = pymem.Pymem("TmForever.exe")

# Addresses (replace with your verified ones)
X_ADDR = 0x1690D470
Y_ADDR = 0x1690D474
Z_ADDR = 0x1690D478
SPEED_ADDR = 0x1665F320
TURN_ADDR = 0x1690D4BC
RACETIME_ADDR = 0x1665F290

# Output file for supervised learning
output_file = "tm_ai_training_data.txt"

# Open CSV for writing
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    # Header (optional)
    writer.writerow(["racetime_ms", "x", "y", "z", "speed", "turning"])

    try:
        while True:
            # Read game state
            racetime = pm.read_int(RACETIME_ADDR)             # ms
            x = pm.read_float(X_ADDR)
            y = pm.read_float(Y_ADDR)
            z = pm.read_float(Z_ADDR)
            speed = pm.read_int(SPEED_ADDR)
            turning = pm.read_float(TURN_ADDR)               # turning in radians

            # Write to CSV
            writer.writerow([racetime, x, y, z, speed, turning])

            # Optional: also print for debugging
            print(f"Time: {racetime} ms | Pos: ({x:.2f},{y:.2f},{z:.2f}) | Speed: {speed:.2f} | Turn: {turning:.2f}")

            # Sample at 20 Hz (50 ms)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Recording stopped by user")
