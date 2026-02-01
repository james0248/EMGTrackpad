import time
from pathlib import Path

import numpy as np
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
from omegaconf import DictConfig, OmegaConf
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


def load_model_config(checkpoint_path: Path) -> DictConfig:
    """Load the model config from the checkpoint's output directory.

    Hydra saves config to .hydra/config.yaml in the output directory.
    """
    config_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at {config_path}. "
            "Expected Hydra config in checkpoint's output directory."
        )
    return OmegaConf.load(config_path)


BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD


class EMGStream:
    """Handles MindRove board connection and data streaming."""

    def __init__(self):
        self.board = None
        self.emg_channels = None

    def connect(self):
        params = MindRoveInputParams()
        self.board = BoardShim(BOARD_ID, params)
        self.board.prepare_session()
        self.board.start_stream(450000)
        self.emg_channels = BoardShim.get_emg_channels(BOARD_ID)

        # Wait for data to start flowing
        time.sleep(1.0)
        if self.board.get_board_data_count() == 0:
            raise RuntimeError("MindRove connected but no data stream.")

    def get_data(self) -> np.ndarray | None:
        if self.board is None:
            return None

        data = self.board.get_board_data()
        if data.size == 0:
            return None

        return data[self.emg_channels]  # (8, n_samples)

    def disconnect(self):
        if self.board is not None and self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()


class RealTimeFilter:
    """Stateful IIR filter that maintains filter state across chunks."""

    def __init__(
        self,
        sample_rate: float,
        num_channels: int,
        notch_freq: float,
        notch_q: float,
        highpass_freq: float,
        filter_order: int,
    ):
        # Build SOS filter (notch + highpass)
        b, a = iirnotch(notch_freq, notch_q, sample_rate)
        sos_notch = tf2sos(b, a)
        sos_highpass = butter(
            filter_order, highpass_freq, btype="highpass", fs=sample_rate, output="sos"
        )
        self.sos = np.vstack([sos_notch, sos_highpass])

        # Initialize filter state for each channel: (n_sections, n_channels, 2)
        zi_single = sosfilt_zi(self.sos)  # (n_sections, 2)
        self.zi = np.tile(zi_single[:, np.newaxis, :], (1, num_channels, 1))

    def filter(self, data: np.ndarray) -> np.ndarray:
        filtered, self.zi = sosfilt(self.sos, data, zi=self.zi, axis=1)
        return filtered


class SlidingBuffer:
    """Maintains a rolling window of EMG samples."""

    def __init__(self, window_samples: int, num_channels: int):
        self.window_samples = window_samples
        self.buffer = np.zeros((num_channels, window_samples), dtype=np.float32)
        self.samples_received = 0

    def append(self, data: np.ndarray):
        n_samples = data.shape[1]
        self.samples_received += n_samples

        if n_samples >= self.window_samples:
            self.buffer = data[:, -self.window_samples :].astype(np.float32)
        else:
            self.buffer = np.roll(self.buffer, -n_samples, axis=1)
            self.buffer[:, -n_samples:] = data.astype(np.float32)

    def is_ready(self) -> bool:
        return self.samples_received >= self.window_samples

    def get_data(self) -> np.ndarray:
        return self.buffer.copy()
