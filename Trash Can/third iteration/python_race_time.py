# Attach to TrackMania 1665F290, 1665F29C, 1665F378, 1690D4A8
import pymem
import time

pm = pymem.Pymem("TmForever.exe")

# Your speed address (HEX)
time_ADDR = 0x1665F290

while True:
    try:
        racetime = pm.read_int(time_ADDR)
        print(f"racetime: {racetime:.2f}")
    except Exception as e:
        print("Read failed:", e)
        break

    time.sleep(0.05)  # 20 Hz
