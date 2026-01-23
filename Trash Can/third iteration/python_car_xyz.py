# Attach to TrackMania value x = 1690D470, 2BB7E070, 2BB7E1D8
# value y = 1690D474, 2478AC68, 281303F8, 28130428, 2BB7E074, 2BB7E128, 2BB7E1DC,
# value z = 1690D478,, 1745E05C, 2478AC6C, 2813006C, 2813009C, 281300CC, 281300FC, 2813012C, 2813015C, 2813018C, 281301BC,
import pymem
import time

# Attach to TrackMania process
pm = pymem.Pymem("TmForever.exe")

# Replace these with the addresses you found via Cheat Engine
X_ADDR = 0x1690D470
Y_ADDR = 0x1690D474
Z_ADDR = 0x1690D478

while True:
    try:
        x = pm.read_float(X_ADDR)
        y = pm.read_float(Y_ADDR)
        z = pm.read_float(Z_ADDR)

        print(f"Position → X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
    except Exception as e:
        print("Read failed:", e)
        break

    time.sleep(0.05)  # 20 Hz
    
    