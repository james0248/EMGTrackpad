import time
from pathlib import Path

import hydra
import numpy as np
import torch
from controller import VirtualTrackpad
from hydra.utils import instantiate
from mindrove.board_shim import BoardShim
from omegaconf import DictConfig
from rich.console import Console
from rich.live import Live
from util.device import get_device

from .streaming import EMGStream, RealTimeFilter, SlidingBuffer, load_model_config
from .visualizer import InferenceState, TerminalVisualizer


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
        self.dxdy_mean = ckpt["dxdy_mean"]  # numpy array (2,)
        self.dxdy_std = ckpt["dxdy_std"]  # numpy array (2,)

        self.num_channels = model_cfg.model.num_channels
        self.emg_sample_rate = model_cfg.model.emg_sample_rate
        self.window_length_s = model_cfg.training.window_length_s
        self.expected_samples = int(self.window_length_s * self.emg_sample_rate)
        self.highpass_freq = model_cfg.preprocessing.highpass_freq

    @torch.no_grad()
    def predict(
        self,
        emg_window: np.ndarray,
        click_threshold: float = 0.5,
    ) -> tuple[float, float, bool, bool, bool]:
        """
        Run inference and return denormalized cursor movement + action states.

        Returns:
            dx, dy: Denormalized cursor movement in pixels
            left_click, right_click, scroll: Action states
        """
        x = torch.from_numpy(emg_window).float().unsqueeze(0).to(self.device)

        output = self.model(x)  # {"dxdy": (1, 2), "actions": (1, 3)}

        # Denormalize dxdy: convert normalized output back to pixel deltas
        # Training normalized as: (dxdy - mean) / std
        # So we reverse: dxdy = normalized * std + mean
        dxdy_normalized = output["dxdy"].cpu().numpy()[0]
        dx = dxdy_normalized[0] * self.dxdy_std[0] + self.dxdy_mean[0]
        dy = dxdy_normalized[1] * self.dxdy_std[1] + self.dxdy_mean[1]

        # Threshold actions (BCEWithLogits -> sigmoid -> threshold)
        actions = torch.sigmoid(output["actions"]).cpu().numpy()[0]
        left_click = actions[0] > click_threshold
        right_click = actions[1] > click_threshold
        scroll = actions[2] > click_threshold

        return float(dx), float(dy), bool(left_click), bool(right_click), bool(scroll)


@hydra.main(version_base=None, config_path="../config/inference", config_name="controller")
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

    console.print(f"Model loaded from: {checkpoint_path}")
    console.print(f"Device: {inference.device}")
    console.print(f"dxdy_mean: {inference.dxdy_mean}")
    console.print(f"dxdy_std: {inference.dxdy_std}")
    console.print("[bold]Connecting to MindRove...[/]")

    try:
        emg_stream.connect()
        console.print("[green]Connected![/]")
        console.print(
            f"Waiting for buffer to fill ({inference.expected_samples} samples)..."
        )

        last_inference_time = 0.0
        current_dx, current_dy = 0.0, 0.0
        current_left_click, current_right_click, current_scroll = False, False, False

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
                    (
                        current_dx,
                        current_dy,
                        current_left_click,
                        current_right_click,
                        current_scroll,
                    ) = inference.predict(window, click_threshold=cfg.click_threshold)
                    visualizer.record_inference()
                    last_inference_time = now

                    # Discard small movements (<= 1px), then scale
                    dx_rounded = round(current_dx)
                    dy_rounded = round(current_dy)
                    if abs(dx_rounded) <= 1:
                        dx_rounded = 0
                    if abs(dy_rounded) <= 1:
                        dy_rounded = 0
                    dx_scaled = round(dx_rounded * cfg.sensitivity)
                    dy_scaled = round(dy_rounded * cfg.sensitivity)
                    scroll_dy = cfg.scroll_amount if current_scroll else 0

                    trackpad.apply(
                        dx=dx_scaled,
                        dy=dy_scaled,
                        is_click_active=current_left_click,
                        scroll_dy=scroll_dy,
                    )

                # Update display
                buffer_fill = min(1.0, buffer.samples_received / buffer.window_samples)
                state = InferenceState(
                    dx=current_dx,
                    dy=current_dy,
                    left_click=current_left_click,
                    right_click=current_right_click,
                    scroll=current_scroll,
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
