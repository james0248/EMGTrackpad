#!/usr/bin/env python3
"""
Analyze and visualize recorded EMG and trackpad data from H5 session files.
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Event code mapping (inverse of track.py)
EVENT_NAMES = {
    1.0: "LEFT_DOWN",
    2.0: "LEFT_UP",
    3.0: "L_DRAG",
    4.0: "RIGHT_DOWN",
    5.0: "RIGHT_UP",
    6.0: "R_DRAG",
    7.0: "SCROLL",
    8.0: "MOVE",
}

EMG_SAMPLING_RATE = 500  # Hz
EMG_HIGHPASS_CUTOFF = 40  # Hz


def highpass_filter(
    data: np.ndarray, cutoff: float = EMG_HIGHPASS_CUTOFF, fs: int = EMG_SAMPLING_RATE
) -> np.ndarray:
    """Apply a 40Hz high-pass Butterworth filter to EMG data.

    Args:
        data: EMG data array of shape (n_samples, n_channels)
        cutoff: High-pass cutoff frequency in Hz
        fs: Sampling frequency in Hz

    Returns:
        Filtered EMG data
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist

    # 4th order Butterworth high-pass filter
    sos = signal.butter(4, normalized_cutoff, btype="high", output="sos")

    # Apply filter to each channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = signal.sosfiltfilt(sos, data[:, ch])

    return filtered


def load_session(filepath: Path) -> dict:
    """Load EMG and trackpad data from an H5 session file."""
    data = {}
    with h5py.File(filepath, "r") as f:
        if "emg" in f:
            data["emg"] = f["emg"][:]
        if "trackpad" in f:
            data["trackpad"] = f["trackpad"][:]
            data["trackpad_cols"] = f["trackpad"].attrs.get(
                "columns", "ts, code, dx, dy"
            )
    return data


def log_trackpad_events(trackpad_data: np.ndarray, max_events: int = 100):
    """Print trackpad events to console."""
    if trackpad_data is None or len(trackpad_data) == 0:
        print("No trackpad data found.")
        return

    print("\n" + "=" * 70)
    print("TRACKPAD EVENT LOG")
    print("=" * 70)
    print(f"{'Time (s)':<12} {'Event':<12} {'dX':>10} {'dY':>10}")
    print("-" * 70)

    # Normalize timestamps to start from 0
    t0 = trackpad_data[0, 0]

    # Group consecutive MOVE events for summary
    move_count = 0
    last_was_move = False

    logged = 0
    for i, row in enumerate(trackpad_data):
        ts, code, dx, dy = row
        event_name = EVENT_NAMES.get(code, f"UNKNOWN({code})")

        # Summarize consecutive MOVE events
        if code == 8.0:  # MOVE
            move_count += 1
            last_was_move = True
            continue
        else:
            if last_was_move and move_count > 0:
                print(f"{'...':<12} {'MOVE':<12} {'':>10} ({move_count} events)")
                logged += 1
                move_count = 0
            last_was_move = False

        print(f"{ts - t0:<12.4f} {event_name:<12} {dx:>10.2f} {dy:>10.2f}")
        logged += 1

        if logged >= max_events:
            remaining = len(trackpad_data) - i - 1
            if remaining > 0:
                print(f"\n... and {remaining} more events")
            break

    # Final move summary
    if move_count > 0:
        print(f"{'...':<12} {'MOVE':<12} {'':>10} ({move_count} events)")

    print("-" * 70)

    # Summary statistics
    total_duration = trackpad_data[-1, 0] - trackpad_data[0, 0]
    print(f"\nTotal events: {len(trackpad_data)}")
    print(f"Duration: {total_duration:.2f} seconds")

    # Event breakdown
    print("\nEvent breakdown:")
    unique, counts = np.unique(trackpad_data[:, 1], return_counts=True)
    for code, count in zip(unique, counts):
        name = EVENT_NAMES.get(code, f"UNKNOWN({code})")
        print(f"  {name}: {count}")


def visualize_emg(emg_data: np.ndarray, sampling_rate: int = EMG_SAMPLING_RATE):
    """Visualize EMG data with time-series plots for each channel."""
    if emg_data is None or len(emg_data) == 0:
        print("No EMG data found.")
        return

    n_samples, n_channels = emg_data.shape
    duration = n_samples / sampling_rate
    time = np.linspace(0, duration, n_samples)

    print(
        f"\nEMG Data: {n_samples} samples, {n_channels} channels, {duration:.2f}s duration"
    )

    # Create figure with subplots for each channel
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_channels))

    for i, ax in enumerate(axes):
        ax.plot(time, emg_data[:, i], color=colors[i], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f"Ch {i + 1}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)

        # Add RMS annotation
        rms = np.sqrt(np.mean(emg_data[:, i] ** 2))
        ax.text(
            0.98,
            0.95,
            f"RMS: {rms:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="gray",
        )

    axes[-1].set_xlabel("Time (seconds)", fontsize=12)
    fig.suptitle("EMG Channels Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def visualize_emg_spectrogram(
    emg_data: np.ndarray, channel: int = 0, sampling_rate: int = EMG_SAMPLING_RATE
):
    """Show spectrogram for a single EMG channel."""
    if emg_data is None or len(emg_data) == 0:
        print("No EMG data found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    n_samples = len(emg_data)
    duration = n_samples / sampling_rate
    time = np.linspace(0, duration, n_samples)

    # Time domain
    ax1.plot(time, emg_data[:, channel], color="#2ecc71", linewidth=0.5)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"EMG Channel {channel + 1} - Time Domain")
    ax1.grid(True, alpha=0.3)

    # Spectrogram
    ax2.specgram(
        emg_data[:, channel], Fs=sampling_rate, cmap="magma", NFFT=256, noverlap=128
    )
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title(f"EMG Channel {channel + 1} - Spectrogram")
    ax2.set_ylim(0, sampling_rate / 2)

    plt.tight_layout()
    plt.show()


def visualize_trackpad_trajectory(trackpad_data: np.ndarray):
    """Visualize trackpad movement as accumulated trajectory."""
    if trackpad_data is None or len(trackpad_data) == 0:
        print("No trackpad data found.")
        return

    # Filter to movement events only (MOVE, DRAG)
    move_mask = np.isin(trackpad_data[:, 1], [3.0, 6.0, 8.0])
    moves = trackpad_data[move_mask]

    if len(moves) == 0:
        print("No movement data found.")
        return

    # Accumulate deltas to get trajectory
    dx = moves[:, 2]
    dy = moves[:, 3]
    x = np.cumsum(dx)
    y = np.cumsum(dy)

    # Normalize time for coloring
    t = moves[:, 0]
    t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create colored scatter based on time
    scatter = ax.scatter(x, y, c=t_norm, cmap="plasma", s=2, alpha=0.6)
    ax.plot(x, y, color="gray", alpha=0.2, linewidth=0.5)

    # Mark start and end
    ax.scatter(
        [x[0]], [y[0]], color="#2ecc71", s=100, zorder=5, label="Start", marker="o"
    )
    ax.scatter(
        [x[-1]], [y[-1]], color="#e74c3c", s=100, zorder=5, label="End", marker="s"
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Time (normalized)")

    ax.invert_yaxis()  # Screen coordinates
    ax.set_xlabel("Accumulated X")
    ax.set_ylabel("Accumulated Y")
    ax.set_title("Trackpad Movement Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        # Try to find the most recent session file in data folder
        data_dir = Path("data")
        session_files = sorted(data_dir.glob("session_*.h5"), reverse=True)
        if not session_files:
            print("Usage: python analyze.py <session_file.h5>")
            print("No session files found in data/ directory.")
            sys.exit(1)
        filepath = session_files[0]
        print(f"Using most recent session: {filepath}")
    else:
        filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    print(f"\nLoading session: {filepath}")
    data = load_session(filepath)

    # Log trackpad events
    if "trackpad" in data:
        log_trackpad_events(data["trackpad"])

    # Visualize trackpad trajectory
    if "trackpad" in data:
        visualize_trackpad_trajectory(data["trackpad"])

    # Visualize EMG
    if "emg" in data:
        emg_raw = data["emg"]
        print(f"\nApplying {EMG_HIGHPASS_CUTOFF}Hz high-pass filter...")
        emg_filtered = highpass_filter(emg_raw)

        visualize_emg(emg_filtered)

        # Show spectrogram for first channel
        if len(emg_filtered) > 0:
            visualize_emg_spectrogram(emg_filtered, channel=0)


if __name__ == "__main__":
    main()
