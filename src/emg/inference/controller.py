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

from emg.inference.output_interface import ControllerModelInterface
from emg.inference.streaming import (
    EMGStream,
    RealTimeFilter,
    SlidingBuffer,
    load_model_config,
)
from emg.inference.trackpad import VirtualTrackpad
from emg.inference.visualizer import InferenceState, TerminalVisualizer
from emg.util.device import get_device

ACTION_NAMES = ["Move", "Scroll", "Left Click", "Right Click"]


class ControllerInference:
    """Loads controller model and runs inference with dxdy denormalization."""

    def __init__(self, checkpoint_path: Path, device: str):
        self.device = get_device(device)

        # Load model config from checkpoint directory
        model_cfg = load_model_config(checkpoint_path)

        # Instantiate model from config
        self.model = instantiate(model_cfg.model)
        self.model.to(self.device)

        # Load checkpoint (includes model weights and dxdy normalization stats)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Load dxdy normalization stats for denormalization
        self.dxdy_mean = ckpt["dxdy_mean"]  # numpy array (4,)
        self.dxdy_std = ckpt["dxdy_std"]  # numpy array (4,)

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


@hydra.main(
    version_base=None, config_path="../config/inference", config_name="controller"
)
def main(cfg: DictConfig):
    console = Console()
    console.print("[bold]Initializing EMG Controller Inference...[/]")

    checkpoint_path = Path(cfg.checkpoint)

    # Initialize components
    inference = ControllerInference(checkpoint_path, cfg.device)
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
    visualizer = TerminalVisualizer(title="EMG Controller Inference")
    trackpad = VirtualTrackpad()
    interface = ControllerModelInterface(
        dxdy_mean=inference.dxdy_mean,
        dxdy_std=inference.dxdy_std,
        sensitivity=cfg.sensitivity,
        click_threshold=cfg.smoothing.click_threshold,
        confidence_threshold=cfg.smoothing.confidence_threshold,
        hold_frames=cfg.smoothing.hold_frames,
        scroll_threshold=cfg.smoothing.scroll_threshold,
        move_threshold=cfg.smoothing.move_threshold,
    )

    console.print(f"Model loaded from: {checkpoint_path}")
    console.print(f"Device: {inference.device}")
    console.print(f"dxdy_mean: {inference.dxdy_mean}")
    console.print(f"dxdy_std: {inference.dxdy_std}")
    console.print(f"Sensitivity: {cfg.sensitivity}")
    console.print(
        f"Action smoothing: threshold={cfg.smoothing.confidence_threshold}, "
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
        current_dx, current_dy = 0.0, 0.0
        current_move, current_left_click = False, False
        current_right_click, current_scroll = False, False
        current_action_probs = np.zeros(4, dtype=np.float64)

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

                    # Get raw values for display
                    (
                        current_dx,
                        current_dy,
                        current_move,
                        current_left_click,
                        current_right_click,
                        current_scroll,
                    ) = interface.get_raw_values(output)
                    current_action_probs = interface.get_action_probs(output)

                    # Process through interface and apply to trackpad
                    command = interface.process(output)
                    command.apply_to(trackpad)

                    visualizer.record_inference()
                    last_inference_time = now

                # Update display
                buffer_fill = min(1.0, buffer.samples_received / buffer.window_samples)
                state = InferenceState(
                    dx=current_dx,
                    dy=current_dy,
                    move=current_move,
                    left_click=current_left_click,
                    right_click=current_right_click,
                    scroll=current_scroll,
                    action_probs=current_action_probs.tolist(),
                    action_names=ACTION_NAMES,
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
