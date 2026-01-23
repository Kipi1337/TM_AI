import pymem
import pymem.process
import time

# Attach to TrackMania
pm = pymem.Pymem("TmForever.exe")

# Your speed address (HEX)
SPEED_ADDR = 0x2B4F3F30

while True:
    try:
        speed = pm.read_float(SPEED_ADDR)
        print(f"Speed: {speed:.2f}")
    except Exception as e:
        print("Read failed:", e)
        break

    time.sleep(0.05)  # 20 Hz
