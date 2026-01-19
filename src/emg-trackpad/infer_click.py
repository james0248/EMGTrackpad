import argparse
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from mindrove.board_shim import BoardIds, BoardShim, MindRoveInputParams
from omegaconf import OmegaConf
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos
from util.device import get_device

BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
CLASS_NAMES = ["nothing", "left", "right"]


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
        """Apply filter to incoming chunk, maintaining state."""
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


class ClickInference:
    """Loads model and runs inference."""

    def __init__(self, config_path: Path, checkpoint_path: Path, device: str):
        self.device = get_device(device)

        # Load config and checkpoint
        cfg = OmegaConf.load(config_path)
        self.model = instantiate(cfg.model)
        self.model.to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.num_channels = cfg.model.num_channels
        self.emg_sample_rate = cfg.model.emg_sample_rate
        self.window_length_s = cfg.training.window_length_s
        self.expected_samples = int(self.window_length_s * self.emg_sample_rate)
        self.highpass_freq = cfg.preprocessing.highpass_freq

    @torch.no_grad()
    def predict(self, emg_window: np.ndarray) -> tuple[int, np.ndarray]:
        x = torch.from_numpy(emg_window).float().unsqueeze(0).to(self.device)

        output = self.model(x)
        logits = output["click"]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = int(logits.argmax(dim=-1).item())

        return pred_class, probs


class TerminalVisualizer:
    """Terminal display for inference results."""

    def __init__(self):
        self.console = Console()
        self.inference_times: list[float] = []
        self.max_times = 100

    def _compute_hz(self) -> float:
        if len(self.inference_times) < 2:
            return 0.0
        elapsed = self.inference_times[-1] - self.inference_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.inference_times) - 1) / elapsed

    def record_inference(self):
        """Record an inference timestamp."""
        self.inference_times.append(time.time())
        if len(self.inference_times) > self.max_times:
            self.inference_times = self.inference_times[-self.max_times :]

    def build_display(
        self,
        pred_class: int,
        probs: np.ndarray,
        buffer_ready: bool,
        buffer_fill: float,
    ) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold")
        table.add_column("Value", min_width=40)

        # Prediction
        pred_name = CLASS_NAMES[pred_class]
        pred_style = "bold green" if pred_class == 0 else "bold red"
        table.add_row("Prediction", f"[{pred_style}]{pred_name.upper()}[/]")

        # Probability bars
        table.add_row("", "")
        for i, name in enumerate(CLASS_NAMES):
            prob = probs[i]
            bar_width = int(prob * 30)
            bar = "[blue]" + "█" * bar_width + "[/]" + "░" * (30 - bar_width)
            table.add_row(f"  {name}", f"{bar} {prob:.1%}")

        # Stats
        table.add_row("", "")
        hz = self._compute_hz()
        table.add_row("Inference Rate", f"{hz:.1f} Hz")

        # Buffer status
        status = (
            "[green]Ready[/]"
            if buffer_ready
            else f"[yellow]Filling {buffer_fill:.0%}[/]"
        )
        table.add_row("Buffer", status)

        return Panel(
            table,
            title="[bold]EMG Click Inference[/]",
            subtitle="Press Ctrl+C to stop",
        )


def main():
    parser = argparse.ArgumentParser(description="Real-time EMG click inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Hydra config (.yaml file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run inference on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=0.1,
        help="Minimum interval between inferences in seconds (default: 0.05 = 20Hz)",
    )
    args = parser.parse_args()

    # Validate paths
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    console = Console()
    console.print("[bold]Initializing EMG Click Inference...[/]")

    # Initialize components
    inference = ClickInference(args.config, args.checkpoint, args.device)
    emg_stream = EMGStream()
    realtime_filter = RealTimeFilter(
        sample_rate=inference.emg_sample_rate,
        num_channels=inference.num_channels,
        notch_freq=60.0,
        notch_q=30.0,
        highpass_freq=inference.highpass_freq,
        filter_order=4,
    )
    buffer = SlidingBuffer(inference.expected_samples, inference.num_channels)
    visualizer = TerminalVisualizer()

    console.print(f"Model loaded from: {args.checkpoint}")
    console.print(f"Device: {inference.device}")
    console.print("[bold]Connecting to MindRove...[/]")

    try:
        emg_stream.connect()
        console.print("[green]Connected![/]")
        console.print(
            f"Waiting for buffer to fill ({inference.expected_samples} samples)..."
        )

        last_inference_time = 0.0
        current_pred = 0
        current_probs = np.array([1.0, 0.0, 0.0])

        with Live(console=console, refresh_per_second=30) as live:
            while True:
                # Poll for new data
                data = emg_stream.get_data()
                if data is not None:
                    filtered = realtime_filter.filter(data)
                    buffer.append(filtered)

                # Run inference at specified interval when buffer is ready
                now = time.time()
                if (
                    buffer.is_ready()
                    and (now - last_inference_time) >= args.inference_interval
                ):
                    window = buffer.get_data()
                    current_pred, current_probs = inference.predict(window)
                    visualizer.record_inference()
                    last_inference_time = now

                # Update display
                buffer_fill = min(1.0, buffer.samples_received / buffer.window_samples)
                panel = visualizer.build_display(
                    current_pred, current_probs, buffer.is_ready(), buffer_fill
                )
                live.update(panel)

                # Small sleep to avoid busy-waiting
                time.sleep(0.01)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/]")
    finally:
        emg_stream.disconnect()
        console.print("[green]Disconnected from MindRove.[/]")


if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()
    main()
