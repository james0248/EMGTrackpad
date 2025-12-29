import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import Quartz
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams

# Constants
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
MOUSE_BUFFER_SIZE = 100

# Event Code Mapping
EVENT_MAP = {
    Quartz.kCGEventLeftMouseDown: 1.0,  # LEFT_DOWN
    Quartz.kCGEventLeftMouseUp: 2.0,  # LEFT_UP
    Quartz.kCGEventLeftMouseDragged: 3.0,  # L_DRAG
    Quartz.kCGEventRightMouseDown: 4.0,  # RIGHT_DOWN
    Quartz.kCGEventRightMouseUp: 5.0,  # RIGHT_UP
    Quartz.kCGEventRightMouseDragged: 6.0,  # R_DRAG
    Quartz.kCGEventScrollWheel: 7.0,  # SCROLL
    Quartz.kCGEventMouseMoved: 8.0,  # MOVE
}


class DataCollector:
    def __init__(self):
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.mouse_buffer = []

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        self.logger = logging.getLogger("Collector")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        self.filepath = data_dir / f"session_{ts}.h5"

    def init_resources(self):
        # MindRove Setup
        self.logger.info("Connecting to MindRove...")
        params = MindRoveInputParams()
        self.board = BoardShim(BOARD_ID, params)
        self.board.prepare_session()
        self.board.start_stream(450000)

        self.emg_channels = BoardShim.get_emg_channels(BOARD_ID)
        sr = BoardShim.get_sampling_rate(BOARD_ID)

        # Check if data is actually flowing
        time.sleep(1.0)
        if self.board.get_board_data_count() == 0:
            raise RuntimeError("MindRove connected but no data stream.")

        with h5py.File(self.filepath, "w") as f:
            # Mouse Dataset
            m_dset = f.create_dataset(
                "trackpad",
                shape=(0, 4),
                maxshape=(None, 4),
                dtype="f8",
                compression="gzip",
            )
            m_dset.attrs["columns"] = "ts, code, dx, dy"

        self.logger.info(f"Ready. Log file: {self.filepath} (SR: {sr}Hz)")

    def save_chunk(self, name, data):
        if len(data) == 0:
            return

        with self.lock:
            with h5py.File(self.filepath, "a") as f:
                if name not in f:
                    # Lazy creation for EMG to match exact column count
                    cols = data.shape[1]
                    f.create_dataset(
                        name,
                        shape=(0, cols),
                        maxshape=(None, cols),
                        dtype="f8",
                        compression="gzip",
                    )

                dset = f[name]
                dset.resize(dset.shape[0] + len(data), axis=0)
                dset[-len(data) :] = data

    def emg_loop(self):
        self.logger.info("EMG Worker started.")
        while not self.stop_event.is_set():
            time.sleep(0.5)

            data = self.board.get_board_data()
            if data.size > 0:
                emg_data = data[self.emg_channels]
                self.save_chunk("emg", emg_data.T)

    def mouse_callback(self, proxy, type_, event, refcon):
        code = EVENT_MAP.get(type_)
        if code is None:
            return event

        # Handle deltas for scroll differently since it uses a different field
        if code == 7.0:  # Scroll
            dx = Quartz.CGEventGetDoubleValueField(
                event, Quartz.kCGScrollWheelEventPointDeltaAxis2
            )
            dy = Quartz.CGEventGetDoubleValueField(
                event, Quartz.kCGScrollWheelEventPointDeltaAxis1
            )
        else:
            dx = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaX)
            dy = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaY)

        self.mouse_buffer.append([time.time(), code, dx, dy])
        if len(self.mouse_buffer) >= MOUSE_BUFFER_SIZE:
            arr = np.array(self.mouse_buffer, dtype=np.float64)
            self.mouse_buffer.clear()
            self.save_chunk("trackpad", arr)

        return event

    def run(self):
        self.init_resources()

        # Start EMG thread
        t = threading.Thread(target=self.emg_loop, daemon=True)
        t.start()

        # Setup Quartz
        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseUp)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDragged)
            | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
        )

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            mask,
            self.mouse_callback,
            self,
        )

        if not tap:
            raise PermissionError("Accessibility permission missing!")

        run_loop_src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(), run_loop_src, Quartz.kCFRunLoopDefaultMode
        )
        Quartz.CGEventTapEnable(tap, True)

        self.logger.info("Collecting... Ctrl+C to stop.")
        try:
            Quartz.CFRunLoopRun()
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
        finally:
            self.cleanup(t)

    def cleanup(self, emg_thread):
        if self.mouse_buffer:
            self.save_chunk("trackpad", np.array(self.mouse_buffer, dtype=np.float64))

        self.stop_event.set()
        emg_thread.join(timeout=2.0)

        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
            self.logger.info("Board released.")


if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()
    try:
        DataCollector().run()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)
