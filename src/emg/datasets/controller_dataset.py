from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from emg.datasets.preprocessing import TARGET_RATE_HZ, preprocess_session


class ControllerSessionDataset(Dataset):
    """PyTorch Dataset for EMG-to-controller from a single session.

    Each sample consists of an EMG window and the corresponding:
        - dxdy: Movement (dx, dy, scroll_dx, scroll_dy)
        - actions: Binary indicators for [move, scroll, left_click, right_click]
    """

    def __init__(
        self,
        filepath: Path,
        window_length_s: float,
        highpass_freq: float,
        emg_scale: float,
        stride_s: float,
        transform: Callable | None = None,
        dxdy_mean: np.ndarray | None = None,
        dxdy_std: np.ndarray | None = None,
        jitter: bool = False,
    ):
        data = preprocess_session(
            filepath, highpass_freq=highpass_freq, emg_scale=emg_scale
        )

        self.emg_timestamps = data["emg_timestamps"]
        self.emg_data = data["emg_data"].astype(np.float32)
        self.emg_sample_rate = data["emg_sample_rate"]
        self.bin_edges = data["bin_edges"]
        self.dxdy = data["dxdy"]
        self.move = data["move"]
        self.left_click = data["left_click"]
        self.right_click = data["right_click"]
        self.scroll = data["scroll"]

        self.window_samples = int(window_length_s * self.emg_sample_rate)
        if stride_s is None:
            stride_s = window_length_s
        self.stride_samples = int(stride_s * self.emg_sample_rate)
        self.max_jitter = self.stride_samples - 1 if jitter else 0
        self.num_windows = (
            len(self.emg_data) - self.window_samples - 2 * self.max_jitter
        ) // self.stride_samples + 1
        self.transform = transform

        # Normalization parameters (set externally after computing global stats)
        self.dxdy_mean = dxdy_mean  # (4,) or None
        self.dxdy_std = dxdy_std  # (4,) or None

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample by index.

        Returns:
            Dictionary containing:
                - emg: EMG window tensor, shape (n_channels, window_samples).
                - dxdy: Normalized movement (dx, dy, scroll_dx, scroll_dy).
                - actions: Binary indicators [move, scroll, left_click, right_click].
        """
        start_idx = self.stride_samples * idx + self.max_jitter
        if self.max_jitter > 0:
            jitter = torch.randint(-self.max_jitter, self.max_jitter + 1, (1,)).item()
            start_idx += jitter
        end_idx = start_idx + self.window_samples
        emg_window = self.emg_data[start_idx:end_idx]
        emg = torch.from_numpy(emg_window.T)

        if self.transform is not None:
            emg = self.transform(emg)

        # Get label from the timestamp at window end
        end_time = self.emg_timestamps[end_idx - 1]
        bin_idx = np.searchsorted(self.bin_edges, end_time, side="right") - 2
        bin_idx = np.clip(bin_idx, 0, len(self.dxdy) - 1)

        # Get dxdy and apply normalization if available
        dxdy = self.dxdy[bin_idx].copy()
        if self.dxdy_mean is not None:
            dxdy = (dxdy - self.dxdy_mean) / (self.dxdy_std + 1e-8)

        # Actions as binary indicators: [move, scroll, left_click, right_click]
        actions = np.array([
            self.move[bin_idx],
            self.scroll[bin_idx],
            self.left_click[bin_idx],
            self.right_click[bin_idx],
        ], dtype=np.float32)

        return {
            "emg": emg,
            "dxdy": torch.from_numpy(dxdy),
            "actions": torch.from_numpy(actions),
        }

    @property
    def left_context(self) -> int:
        return self.window_samples

    @property
    def output_rate_hz(self) -> int:
        return TARGET_RATE_HZ


def make_controller_dataset(
    filepaths: list[Path],
    window_length_s: float = 0.2,
    highpass_freq: float = 40.0,
    emg_scale: float = 1.0,
    stride_s: float | None = None,
    transform=None,
    jitter: bool = False,
) -> ConcatDataset:
    """Create a concatenated controller dataset from multiple session files.

    Computes global dxdy normalization statistics across all sessions and
    applies them to each session dataset.

    Returns:
        ConcatDataset with additional attributes:
            - dxdy_mean: Global mean of dxdy values (4,) as [dx, dy, scroll_dx, scroll_dy]
            - dxdy_std: Global std of dxdy values (4,) as [dx, dy, scroll_dx, scroll_dy]
    """
    # First pass: create datasets without normalization
    datasets = [
        ControllerSessionDataset(
            fp,
            window_length_s=window_length_s,
            highpass_freq=highpass_freq,
            emg_scale=emg_scale,
            stride_s=stride_s,
            transform=transform,
            jitter=jitter,
        )
        for fp in filepaths
    ]

    # Compute global normalization statistics
    all_dxdy = np.concatenate([ds.dxdy for ds in datasets], axis=0)
    dxdy_mean = all_dxdy.mean(axis=0).astype(np.float32)
    dxdy_std = all_dxdy.std(axis=0).astype(np.float32)

    # Apply normalization parameters to all datasets
    for ds in datasets:
        ds.dxdy_mean = dxdy_mean
        ds.dxdy_std = dxdy_std

    concat = ConcatDataset(datasets)
    # Store normalization stats on concat dataset for checkpoint saving
    concat.dxdy_mean = dxdy_mean
    concat.dxdy_std = dxdy_std

    return concat
