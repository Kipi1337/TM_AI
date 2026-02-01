from pynput.keyboard import Key, Controller
import time

# =========================
# CONFIG
# =========================
ACTION_TEST = True
ACTION_DELAY = 0.4
RUNNING = True

keyboard = Controller()

# =========================
# ACTION HANDLER
# =========================
def release_all():
    # pynput automatically releases keys after pressing if we use 'press' + 'release'
    pass  # no-op because we manually release in apply_action


def apply_action(action):
    # Release all first
    keyboard.release(Key.up)
    keyboard.release(Key.down)
    keyboard.release(Key.left)
    keyboard.release(Key.right)

    # Press keys according to action
    if action == 0:
        print("UP")
        keyboard.press(Key.up)
    elif action == 1:
        print("DOWN")
        keyboard.press(Key.down)
    elif action == 2:
        print("LEFT")
        keyboard.press(Key.left)
    elif action == 3:
        print("RIGHT")
        keyboard.press(Key.right)
    elif action == 4:
        print("UP + LEFT")
        keyboard.press(Key.up)
        keyboard.press(Key.left)
    elif action == 5:
        print("UP + RIGHT")
        keyboard.press(Key.up)
        keyboard.press(Key.right)
    elif action == 6:
        print("NO INPUT")

    # Small delay to simulate hold
    time.sleep(ACTION_DELAY)

    # Release pressed keys after delay
    keyboard.release(Key.up)
    keyboard.release(Key.down)
    keyboard.release(Key.left)
    keyboard.release(Key.right)


# =========================
# MAIN LOOP
# =========================
print("Arrow key test with pynput")
print("→ Focus Trackmania")
print("→ Press Ctrl+C to stop\n")

try:
    while True:
        if ACTION_TEST:
            for action in range(7):
                apply_action(action)
        else:
            time.sleep(0.5)
except KeyboardInterrupt:
    release_all()
    print("Clean exit")
