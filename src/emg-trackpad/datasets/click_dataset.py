from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .preprocessing import TARGET_RATE_HZ, preprocess_session


class ClickSessionDataset(Dataset):
    """PyTorch Dataset for EMG-to-click classification from a single session.

    Each sample consists of an EMG window and the corresponding click class:
        0 = nothing (no button pressed)
        1 = left click
        2 = right click
    """

    def __init__(
        self,
        filepath: Path,
        window_length_s: float,
        highpass_freq: float,
        emg_scale: float,
        transform: Callable | None = None,
    ):
        data = preprocess_session(
            filepath, highpass_freq=highpass_freq, emg_scale=emg_scale
        )

        self.emg_timestamps = data["emg_timestamps"]
        self.emg_data = data["emg_data"].astype(np.float32)
        self.bin_edges = data["bin_edges"]
        self.n_bins = len(self.bin_edges) - 1
        self.left_click = data["left_click"]
        self.right_click = data["right_click"]

        self.window_duration = window_length_s
        self.window_samples = int(self.window_duration * data["emg_sample_rate"])
        self.transform = transform

        self.valid_indices = self._compute_valid_indices()

    def _get_emg_window_indices(self, bin_end_time: float) -> tuple[int, int]:
        """Get EMG array indices for the window ending at bin_end_time."""
        end_idx = np.searchsorted(self.emg_timestamps, bin_end_time)
        start_idx = end_idx - self.window_samples
        return start_idx, end_idx

    def _compute_valid_indices(self) -> np.ndarray:
        """Find bin indices that have enough EMG history for a full window."""
        valid = []
        for i in range(self.n_bins):
            bin_end_time = self.bin_edges[i + 1]
            window_start_time = bin_end_time - self.window_duration
            start_idx = np.searchsorted(self.emg_timestamps, window_start_time)
            end_idx = np.searchsorted(self.emg_timestamps, bin_end_time)

            if end_idx - start_idx >= self.window_samples:
                valid.append(i)

        return np.array(valid)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample by index.

        Returns:
            Dictionary containing:
                - emg: EMG window tensor, shape (n_channels, window_samples).
                - click: Click class (0=nothing, 1=left, 2=right).
        """
        bin_idx = self.valid_indices[idx]
        bin_end_time = self.bin_edges[bin_idx + 1]
        start_idx, end_idx = self._get_emg_window_indices(bin_end_time)

        emg_window = self.emg_data[start_idx:end_idx]
        emg = torch.from_numpy(emg_window.T)

        if self.transform is not None:
            emg = self.transform(emg)

        # Convert to class index: 0=nothing, 1=left, 2=right
        if self.right_click[bin_idx]:
            click = 2
        elif self.left_click[bin_idx]:
            click = 1
        else:
            click = 0

        return {
            "emg": emg,
            "click": torch.tensor(click, dtype=torch.long),
        }

    @property
    def left_context(self) -> int:
        return self.window_samples

    @property
    def output_rate_hz(self) -> int:
        return TARGET_RATE_HZ


def make_click_dataset(
    filepaths: list[Path],
    window_length_s: float = 0.2,
    highpass_freq: float = 40.0,
    emg_scale: float = 1.0,
    transform=None,
) -> ConcatDataset:
    """Create a concatenated click dataset from multiple session files."""
    datasets = [
        ClickSessionDataset(
            fp,
            window_length_s=window_length_s,
            highpass_freq=highpass_freq,
            emg_scale=emg_scale,
            transform=transform,
        )
        for fp in filepaths
    ]

    return ConcatDataset(datasets)
