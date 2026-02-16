from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

TARGET_RATE_HZ = 50
DEFAULT_NOTCH_FREQ = 60.0
DEFAULT_NOTCH_Q = 30.0
BIN_DURATION_S = 1.0 / TARGET_RATE_HZ

LEFT_DOWN = 1.0
LEFT_UP = 2.0
L_DRAG = 3.0
RIGHT_DOWN = 4.0
RIGHT_UP = 5.0
R_DRAG = 6.0
SCROLL = 7.0
MOVE = 8.0


def filter_emg(
    emg_data: np.ndarray,
    sample_rate: float,
    highpass_freq: float,
    filter_order: int,
    notch_freq: float = DEFAULT_NOTCH_FREQ,
    notch_q: float = DEFAULT_NOTCH_Q,
) -> np.ndarray:
    """Apply notch and high-pass filters to EMG data.

    Args:
        emg_data: EMG signal array of shape (n_samples, n_channels).
        sample_rate: Sampling rate of the EMG data in Hz.
        highpass_freq: Cutoff frequency for the high-pass filter in Hz.
        filter_order: Order of the Butterworth filter.
        notch_freq: Center frequency for notch filter (default 60Hz). None to disable.
        notch_q: Quality factor for notch filter (higher = narrower notch).

    Returns:
        Filtered EMG data with the same shape as the input.
    """

    # 60Hz notch filter and high-pass filter
    b, a = iirnotch(notch_freq, notch_q, sample_rate)
    sos_notch = tf2sos(b, a)
    sos_highpass = butter(
        filter_order, highpass_freq, btype="highpass", fs=sample_rate, output="sos"
    )

    sos = np.vstack([sos_notch, sos_highpass])
    return sosfilt(sos, emg_data, axis=0)


def load_session(filepath: Path) -> dict:
    """Load a recording session from an HDF5 file.

    Args:
        filepath: Path to the HDF5 session file.

    Returns:
        Dictionary containing:
            - emg_sample_rate: Sampling rate of the EMG data in Hz.
            - emg_timestamps: Array of timestamps for each EMG sample.
            - emg_data: EMG signal array of shape (n_samples, n_channels).
            - trackpad: Trackpad event array with columns [timestamp, code, dx, dy].
    """
    with h5py.File(filepath, "r") as f:
        emg_sample_rate = f.attrs["emg_sample_rate_hz"]
        emg = f["emg"][:]
        trackpad = f["trackpad"][:]

    return {
        "emg_sample_rate": int(emg_sample_rate),
        "emg_timestamps": emg[:, 0],
        "emg_data": emg[:, 1:],
        "trackpad": trackpad,
    }


def build_time_bins(start_time: float, end_time: float) -> np.ndarray:
    """Create time bin edges for downsampling to the target rate."""

    n_bins = int(np.ceil((end_time - start_time) * TARGET_RATE_HZ))
    return start_time + np.arange(n_bins + 1) * BIN_DURATION_S


def bin_trackpad_events(trackpad: np.ndarray, bin_edges: np.ndarray) -> dict:
    """Aggregate trackpad events into time bins.

    Accumulates cursor movement (dx, dy) and scroll movement (scroll_dx, scroll_dy)
    within each bin. Tracks button states (left click, right click), scroll events,
    and move events.

    Event type separation:
        - MOVE events: accumulate dx/dy to cursor columns (0-1)
        - SCROLL events: accumulate dx/dy to scroll columns (2-3)
        - DRAG events (L_DRAG, R_DRAG): treated like MOVE for cursor dx/dy

    Args:
        trackpad: Trackpad event array with columns [timestamp, code, dx, dy].
        bin_edges: Array of time bin edge timestamps.

    Returns:
        Dictionary containing:
            - dxdy: Movement per bin, shape (n_bins, 4) as [dx, dy, scroll_dx, scroll_dy].
            - move: Binary move flag per bin, shape (n_bins,).
            - left_click: Binary left button state per bin, shape (n_bins,).
            - right_click: Binary right button state per bin, shape (n_bins,).
            - scroll: Binary scroll flag per bin, shape (n_bins,).
    """
    n_bins = len(bin_edges) - 1
    dxdy = np.zeros((n_bins, 4), dtype=np.float32)
    move = np.zeros(n_bins, dtype=np.float32)
    left_click = np.zeros(n_bins, dtype=np.float32)
    right_click = np.zeros(n_bins, dtype=np.float32)
    scroll = np.zeros(n_bins, dtype=np.float32)

    ts = trackpad[:, 0]
    codes = trackpad[:, 1]

    bin_indices = np.searchsorted(bin_edges, ts, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Separate cursor movement (MOVE, L_DRAG, R_DRAG) from scroll movement
    cursor_mask = (codes == MOVE) | (codes == L_DRAG) | (codes == R_DRAG)
    scroll_mask = codes == SCROLL

    # Accumulate cursor dx/dy (columns 0-1) for MOVE and DRAG events
    if cursor_mask.any():
        cursor_indices = bin_indices[cursor_mask]
        np.add.at(dxdy, (cursor_indices, 0), trackpad[cursor_mask, 2])
        np.add.at(dxdy, (cursor_indices, 1), trackpad[cursor_mask, 3])

    # Accumulate scroll dx/dy (columns 2-3) for SCROLL events
    if scroll_mask.any():
        scroll_indices = bin_indices[scroll_mask]
        np.add.at(dxdy, (scroll_indices, 2), trackpad[scroll_mask, 2])
        np.add.at(dxdy, (scroll_indices, 3), trackpad[scroll_mask, 3])
        scroll[scroll_indices] = 1.0

    # Mark move events (MOVE or DRAG with nonzero dx/dy)
    cursor_events_with_motion = cursor_mask & (
        (trackpad[:, 2] != 0) | (trackpad[:, 3] != 0)
    )
    if cursor_events_with_motion.any():
        move[bin_indices[cursor_events_with_motion]] = 1.0

    # Track button states across bins
    current_left = False
    current_right = False
    for bin_idx in range(n_bins):
        bin_codes = codes[bin_indices == bin_idx]
        if len(bin_codes) > 0:
            if LEFT_DOWN in bin_codes:
                current_left = True
            if LEFT_UP in bin_codes:
                current_left = False
            if RIGHT_DOWN in bin_codes:
                current_right = True
            if RIGHT_UP in bin_codes:
                current_right = False
        if current_left:
            left_click[bin_idx] = 1.0
        if current_right:
            right_click[bin_idx] = 1.0

    return {
        "dxdy": dxdy,
        "move": move,
        "left_click": left_click,
        "right_click": right_click,
        "scroll": scroll,
    }


def preprocess_session(filepath: Path, highpass_freq: float, emg_scale: float) -> dict:
    """Load and preprocess a recording session for model training.

    Applies high-pass filtering and scaling to EMG data, then bins trackpad
    events to the target rate (50 Hz).

    Args:
        filepath: Path to the HDF5 session file.
        highpass_freq: High-pass cutoff frequency in Hz.
        emg_scale: Scale factor applied to filtered EMG data.

    Returns:
        Dictionary containing:
            - emg_sample_rate: Original EMG sampling rate in Hz.
            - emg_timestamps: Array of EMG sample timestamps.
            - emg_data: Filtered and scaled EMG data.
            - bin_edges: Time bin edge timestamps at target rate (n_bins + 1).
            - dxdy: Movement targets per bin, shape (n_bins, 4) as [dx, dy, scroll_dx, scroll_dy].
            - move: Binary move flag per bin, shape (n_bins,).
            - left_click: Binary left button state per bin, shape (n_bins,).
            - right_click: Binary right button state per bin, shape (n_bins,).
            - scroll: Binary scroll flag per bin, shape (n_bins,).
    """
    session = load_session(filepath)

    emg_ts = session["emg_timestamps"]
    emg_data = session["emg_data"]
    trackpad = session["trackpad"]

    # Apply EMG preprocessing: high-pass filter + scaling
    emg_data = filter_emg(
        emg_data,
        sample_rate=session["emg_sample_rate"],
        highpass_freq=highpass_freq,
        filter_order=4,
        notch_freq=DEFAULT_NOTCH_FREQ,
        notch_q=DEFAULT_NOTCH_Q,
    )
    emg_data = emg_data * emg_scale

    start_time = max(emg_ts[0], trackpad[0, 0])
    end_time = min(emg_ts[-1], trackpad[-1, 0])
    bin_edges = build_time_bins(start_time, end_time)

    targets = bin_trackpad_events(trackpad, bin_edges)

    return {
        "emg_sample_rate": session["emg_sample_rate"],
        "emg_timestamps": emg_ts,
        "emg_data": emg_data,
        "bin_edges": bin_edges,
        "dxdy": targets["dxdy"],
        "move": targets["move"],
        "left_click": targets["left_click"],
        "right_click": targets["right_click"],
        "scroll": targets["scroll"],
    }
