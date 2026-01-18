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
        stride_s: float,
        transform: Callable | None = None,
    ):
        data = preprocess_session(
            filepath, highpass_freq=highpass_freq, emg_scale=emg_scale
        )

        self.emg_timestamps = data["emg_timestamps"]
        self.emg_data = data["emg_data"].astype(np.float32)
        self.emg_sample_rate = data["emg_sample_rate"]
        self.bin_edges = data["bin_edges"]
        self.left_click = data["left_click"]
        self.right_click = data["right_click"]

        self.window_samples = int(window_length_s * self.emg_sample_rate)
        self.stride_samples = int(stride_s * self.emg_sample_rate)
        self.num_windows = (
            len(self.emg_data) - self.window_samples
        ) // self.stride_samples + 1
        self.transform = transform

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample by index.

        Returns:
            Dictionary containing:
                - emg: EMG window tensor, shape (n_channels, window_samples).
                - click: Click class (0=nothing, 1=left, 2=right).
        """
        start_idx = self.stride_samples * idx
        end_idx = start_idx + self.window_samples
        emg_window = self.emg_data[start_idx:end_idx]
        emg = torch.from_numpy(emg_window.T)

        if self.transform is not None:
            emg = self.transform(emg)

        # Get label from the timestamp at window end
        end_time = self.emg_timestamps[end_idx - 1]
        bin_idx = np.searchsorted(self.bin_edges, end_time, side="right") - 2
        bin_idx = np.clip(bin_idx, 0, len(self.left_click) - 1)

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
    stride_s: float | None = None,
    transform=None,
) -> ConcatDataset:
    """Create a concatenated click dataset from multiple session files."""
    datasets = [
        ClickSessionDataset(
            fp,
            window_length_s=window_length_s,
            highpass_freq=highpass_freq,
            emg_scale=emg_scale,
            stride_s=stride_s,
            transform=transform,
        )
        for fp in filepaths
    ]

    return ConcatDataset(datasets)
