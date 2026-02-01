import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from mindrove.board_shim import BoardShim
from omegaconf import DictConfig
from rich.console import Console
from rich.live import Live
from util.device import get_device

from .output_interface import ClickModelInterface
from .streaming import EMGStream, RealTimeFilter, SlidingBuffer, load_model_config
from .trackpad import VirtualTrackpad
from .visualizer import InferenceState, TerminalVisualizer

CLASS_NAMES = ["nothing", "left", "right"]


class ClickInference:
    """Loads model and runs inference."""

    def __init__(self, checkpoint_path: Path, device: str):
        self.device = get_device(device)

        # Load model config from checkpoint directory
        model_cfg = load_model_config(checkpoint_path)

        # Instantiate model from config
        self.model = instantiate(model_cfg.model)
        self.model.to(self.device)

        # Load checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.num_channels = model_cfg.model.num_channels
        self.emg_sample_rate = model_cfg.model.emg_sample_rate
        self.window_length_s = model_cfg.training.window_length_s
        self.expected_samples = int(self.window_length_s * self.emg_sample_rate)
        self.highpass_freq = model_cfg.preprocessing.highpass_freq

    @torch.no_grad()
    def forward(self, emg_window: np.ndarray) -> dict[str, torch.Tensor]:
        """Run model forward pass and return raw output."""
        x = torch.from_numpy(emg_window).float().unsqueeze(0).to(self.device)
        return self.model(x)


@hydra.main(version_base=None, config_path="../config/inference", config_name="click")
def main(cfg: DictConfig):
    console = Console()
    console.print("[bold]Initializing EMG Click Inference...[/]")

    checkpoint_path = Path(cfg.checkpoint)

    # Initialize components
    inference = ClickInference(checkpoint_path, cfg.device)
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
    visualizer = TerminalVisualizer(title="EMG Click Inference")
    trackpad = VirtualTrackpad()
    interface = ClickModelInterface(
        confidence_threshold=cfg.smoothing.confidence_threshold,
        hold_frames=cfg.smoothing.hold_frames,
    )

    console.print(f"Model loaded from: {checkpoint_path}")
    console.print(f"Device: {inference.device}")
    console.print(
        f"Smoothing: confidence_threshold={cfg.smoothing.confidence_threshold}, "
        f"hold_frames={cfg.smoothing.hold_frames}"
    )
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
        current_command = None

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
                    and (now - last_inference_time) >= cfg.inference_interval
                ):
                    window = buffer.get_data()
                    output = inference.forward(window)
                    current_pred, current_probs = interface.get_prediction_info(output)
                    current_command = interface.process(output)
                    current_command.apply_to(trackpad)
                    visualizer.record_inference()
                    last_inference_time = now

                # Update display
                buffer_fill = min(1.0, buffer.samples_received / buffer.window_samples)
                smoothed_left = (
                    current_command.is_click_active if current_command else False
                )
                state = InferenceState(
                    prediction=current_pred,
                    probs=current_probs.tolist(),
                    class_names=CLASS_NAMES,
                    left_click=smoothed_left,
                    right_click=(current_pred == 2),
                    buffer_ready=buffer.is_ready(),
                    buffer_fill=buffer_fill,
                )
                panel = visualizer.build_display(state)
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
