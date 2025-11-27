import pyautogui
import time
import csv

fps = 50
interval = 1 / fps
filename = "mouse_log_50fps.csv"

print(f"Target: {fps} FPS (interval: {interval:.4f}s)")
print("Recording started. Press Ctrl+C to stop.")

try:
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "X", "Y"])

        next_capture_time = time.time()

        while True:
            x, y = pyautogui.position()
            current_time = time.time()
            writer.writerow([current_time, x, y])

            next_capture_time += interval

            sleep_time = next_capture_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nRecording stopped.")
