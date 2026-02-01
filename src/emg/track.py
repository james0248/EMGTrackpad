import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import Quartz
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams

# Constants
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
MOUSE_BUFFER_SIZE = 100

# Event types and their codes
SCROLL_EVENT_CODE = 7.0
EVENT_TYPES = [
    (Quartz.kCGEventLeftMouseDown, 1.0),
    (Quartz.kCGEventLeftMouseUp, 2.0),
    (Quartz.kCGEventLeftMouseDragged, 3.0),
    (Quartz.kCGEventRightMouseDown, 4.0),
    (Quartz.kCGEventRightMouseUp, 5.0),
    (Quartz.kCGEventRightMouseDragged, 6.0),
    (Quartz.kCGEventScrollWheel, SCROLL_EVENT_CODE),
    (Quartz.kCGEventMouseMoved, 8.0),
]
EVENT_MAP = {event_type: code for event_type, code in EVENT_TYPES}


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
        self.emg_sample_rate = BoardShim.get_sampling_rate(BOARD_ID)
        self.timestamp_channel = BoardShim.get_timestamp_channel(BOARD_ID)

        time.sleep(1.0)
        if self.board.get_board_data_count() == 0:
            raise RuntimeError("MindRove connected but no data stream.")

        with h5py.File(self.filepath, "w") as f:
            f.attrs["emg_sample_rate_hz"] = self.emg_sample_rate
            f.create_dataset(
                "trackpad",
                shape=(0, 4),
                maxshape=(None, 4),
                dtype="f8",
                compression="gzip",
            )

        self.logger.info(
            f"Ready. Log file: {self.filepath} "
            f"(EMG SR: {self.emg_sample_rate}Hz, {len(self.emg_channels)} channels)"
        )

    def save_chunk(self, name, data):
        if not len(data):
            return

        with self.lock:
            with h5py.File(self.filepath, "a") as f:
                if name not in f:
                    f.create_dataset(
                        name,
                        shape=(0, data.shape[1]),
                        maxshape=(None, data.shape[1]),
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
            if data.size:
                emg_data = data[self.emg_channels]
                timestamps = data[self.timestamp_channel]
                emg_with_ts = np.column_stack([timestamps, emg_data.T])
                self.save_chunk("emg", emg_with_ts)

    def mouse_callback(self, proxy, type_, event, refcon):
        code = EVENT_MAP.get(type_)
        if code is None:
            return event

        if code == SCROLL_EVENT_CODE:
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
            self.save_chunk("trackpad", np.array(self.mouse_buffer, dtype=np.float64))
            self.mouse_buffer.clear()

        return event

    def run(self):
        self.init_resources()

        emg_thread = threading.Thread(target=self.emg_loop, daemon=True)
        emg_thread.start()

        event_mask = sum(
            Quartz.CGEventMaskBit(event_type) for event_type, _ in EVENT_TYPES
        )
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            event_mask,
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
            self.cleanup(emg_thread)

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
