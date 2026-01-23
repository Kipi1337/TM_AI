import pymem
import time

# Attach to TrackMania
pm = pymem.Pymem("TmForever.exe")

# Absolute memory address (session-specific)
ADDR = 0x1665F320  # your 4-byte int address

while True:
    try:
        value = pm.read_int(ADDR)  # read 4 bytes as int
        print(f"Value: {value}")
    except Exception as e:
        print("Read failed:", e)
        break
    time.sleep(0.05)