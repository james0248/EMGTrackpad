import os
import sys
import time

import h5py
import numpy as np
import Quartz

FILENAME = "mouse_log.h5"
BUFFER_SIZE = 100  # Flush every BUFFER_SIZE events

# Map event codes to numbers
EVT_LEFT_DOWN = 1.0
EVT_LEFT_UP = 2.0
EVT_L_DRAG = 3.0
EVT_RIGHT_DOWN = 4.0
EVT_RIGHT_UP = 5.0
EVT_R_DRAG = 6.0
EVT_SCROLL = 7.0
EVT_MOVE = 8.0

data_buffer = []


def init_storage():
    if not os.path.exists(FILENAME):
        with h5py.File(FILENAME, "w") as f:
            # maxshape=(None, 4) means the dataset can grow
            dset = f.create_dataset(
                "mouse_events",
                shape=(0, 4),
                maxshape=(None, 4),
                dtype=np.float64,
                chunks=True,
                compression="gzip",
            )

            dset.attrs["columns"] = "Timestamp, EventCode, dx, dy"
            dset.attrs["codes"] = (
                "1:L_DOWN, 2:L_UP, 3:L_DRAG, "
                "4:R_DOWN, 5:R_UP, 6:R_DRAG, "
                "7:SCROLL, 8:MOVE"
            )
            print(f"New file created: {FILENAME}")


def flush_buffer():
    global data_buffer
    if not data_buffer:
        return

    try:
        new_data = np.array(data_buffer, dtype=np.float64)

        with h5py.File(FILENAME, "a") as f:  # 'a' 모드: 수정/추가
            dset = f["mouse_events"]

            current_len = dset.shape[0]
            add_len = new_data.shape[0]

            dset.resize(current_len + add_len, axis=0)
            dset[current_len:] = new_data

        print(
            f" >> [Auto-Save] {len(data_buffer)} events saved. (Total: {current_len + add_len} events)"
        )
        data_buffer = []

    except Exception as e:
        print(f"Error saving: {e}")


def mouse_event_callback(proxy, event_type, event, refcon):
    raw_ts = time.perf_counter()
    ts_str = f"{raw_ts:.6f}"

    dx = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaX)
    dy = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaY)

    code = 0.0
    save_dx, save_dy = 0.0, 0.0

    # 1. Left Mouse
    if event_type == Quartz.kCGEventLeftMouseDown:
        code = EVT_LEFT_DOWN
        print(f"[{ts_str}] LEFT DOWN  | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventLeftMouseUp:
        code = EVT_LEFT_UP
        print(f"[{ts_str}] LEFT UP    | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventLeftMouseDragged:
        if dx != 0 or dy != 0:
            code = EVT_L_DRAG
            save_dx, save_dy = dx, dy
            print(f"[{ts_str}] L-DRAGGING | dx: {dx:10.6f}, dy: {dy:10.6f}")

    # 2. Right Mouse
    elif event_type == Quartz.kCGEventRightMouseDown:
        code = EVT_RIGHT_DOWN
        print(f"[{ts_str}] RIGHT DOWN | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventRightMouseUp:
        code = EVT_RIGHT_UP
        print(f"[{ts_str}] RIGHT UP   | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventRightMouseDragged:
        if dx != 0 or dy != 0:
            code = EVT_R_DRAG
            save_dx, save_dy = dx, dy
            print(f"[{ts_str}] R-DRAGGING | dx: {dx:10.6f}, dy: {dy:10.6f}")

    # 3. Scroll
    elif event_type == Quartz.kCGEventScrollWheel:
        s_dx = Quartz.CGEventGetDoubleValueField(
            event, Quartz.kCGScrollWheelEventPointDeltaAxis2
        )
        s_dy = Quartz.CGEventGetDoubleValueField(
            event, Quartz.kCGScrollWheelEventPointDeltaAxis1
        )
        if s_dy != 0 or s_dx != 0:
            code = EVT_SCROLL
            save_dx, save_dy = s_dx, s_dy
            print(f"[{ts_str}] SCROLL     | dx: {s_dx:10.6f}, dy: {s_dy:10.6f}")

    # 4. Move
    elif event_type == Quartz.kCGEventMouseMoved:
        if dx != 0 or dy != 0:
            code = EVT_MOVE
            save_dx, save_dy = dx, dy
            print(f"[{ts_str}] MOVE       | dx: {dx:10.6f}, dy: {dy:10.6f}")

    if code != 0.0:
        data_buffer.append([raw_ts, code, save_dx, save_dy])
        if len(data_buffer) >= BUFFER_SIZE:
            flush_buffer()

    return event


def run_event_capture():
    init_storage()

    event_mask = (
        Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseUp)
        | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDragged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
    )

    event_tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        event_mask,
        mouse_event_callback,
        None,
    )

    if not event_tap:
        print("Error: Accessibility permissions required.")
        sys.exit(1)

    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
    loop = Quartz.CFRunLoopGetCurrent()
    Quartz.CFRunLoopAddSource(loop, run_loop_source, Quartz.kCFRunLoopDefaultMode)
    Quartz.CGEventTapEnable(event_tap, True)

    print(
        f"Quartz Incremental Save Tracker Running... (Auto-save every {BUFFER_SIZE} events)"
    )
    print(f"Logging to: {FILENAME}")

    try:
        Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        print("\nStopping...")
        flush_buffer()
        print("Exit.")


if __name__ == "__main__":
    run_event_capture()
