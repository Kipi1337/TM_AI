# Attach to TrackMania 1690D4BC
# 1690D4C0
# 2C2636F0
# 2C263998
# 2C263A40
# 2C263AE8
# 2C263B90
# 2C263C80
import pymem
import time

pm = pymem.Pymem("TmForever.exe")

# Your speed address (HEX)
turning_rad = 0x1690D4BC

while True:
    try:
        turning = pm.read_int(turning_rad)
        print(f"turning: {turning:.2f}")
    except Exception as e:
        print("Read failed:", e)
        break

    time.sleep(0.05)  # 20 Hz
