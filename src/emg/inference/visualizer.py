import time
from dataclasses import dataclass

from rich.panel import Panel
from rich.table import Table


@dataclass
class InferenceState:
    """Unified state for display."""

    # Click model fields (None if not applicable)
    prediction: int | None = None  # Class index (0=nothing, 1=left, 2=right)
    probs: list[float] | None = None  # Probabilities for each class
    class_names: list[str] | None = None  # ["nothing", "left", "right"]

    # Controller model fields (None if not applicable)
    dx: float | None = None
    dy: float | None = None

    # Shared action states
    move: bool = False
    left_click: bool = False
    right_click: bool = False
    scroll: bool = False

    # Stats (always present)
    buffer_ready: bool = False
    buffer_fill: float = 0.0


class TerminalVisualizer:
    """Unified terminal display for inference results."""

    def __init__(self, title: str = "EMG Inference"):
        self.title = title
        self.inference_times: list[float] = []
        self.max_times = 100

    def _compute_hz(self) -> float:
        if len(self.inference_times) < 2:
            return 0.0
        elapsed = self.inference_times[-1] - self.inference_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.inference_times) - 1) / elapsed

    def record_inference(self) -> None:
        """Record an inference timestamp."""
        self.inference_times.append(time.time())
        if len(self.inference_times) > self.max_times:
            self.inference_times = self.inference_times[-self.max_times :]

    def _format_action_row(self, label: str, active: bool) -> tuple[str, str]:
        """Format an action state row with appropriate styling."""
        style = "bold green" if active else "dim"
        text = "ACTIVE" if active else "inactive"
        return label, f"[{style}]{text}[/]"

    def _format_value_row(
        self, label: str, value: float | None, fmt: str
    ) -> tuple[str, str]:
        """Format a numeric value row with fallback for None."""
        if value is not None:
            return label, fmt.format(value)
        return label, "[dim]—[/]"

    def build_display(self, state: InferenceState) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold")
        table.add_column("Value", min_width=40)

        # Prediction section (click model)
        if state.prediction is not None and state.class_names is not None:
            pred_name = state.class_names[state.prediction]
            pred_style = "bold green" if state.prediction == 0 else "bold red"
            table.add_row("Prediction", f"[{pred_style}]{pred_name.upper()}[/]")
        else:
            table.add_row("Prediction", "[dim]—[/]")

        # Probability bars (click model)
        table.add_row("", "")
        if state.probs is not None and state.class_names is not None:
            for i, name in enumerate(state.class_names):
                prob = state.probs[i]
                bar_width = int(prob * 30)
                bar = "[blue]" + "█" * bar_width + "[/]" + "░" * (30 - bar_width)
                table.add_row(f"  {name}", f"{bar} {prob:.1%}")
        else:
            table.add_row("  [dim]N/A[/]", "")

        # Cursor movement section (controller model)
        table.add_row("", "")
        table.add_row(*self._format_value_row("Cursor dx", state.dx, "{:+.2f} px"))
        table.add_row(*self._format_value_row("Cursor dy", state.dy, "{:+.2f} px"))

        # Action states (shared)
        table.add_row("", "")
        table.add_row(*self._format_action_row("Move", state.move))
        table.add_row(*self._format_action_row("Left Click", state.left_click))
        table.add_row(*self._format_action_row("Right Click", state.right_click))
        table.add_row(*self._format_action_row("Scroll", state.scroll))

        # Stats
        table.add_row("", "")
        table.add_row("Inference Rate", f"{self._compute_hz():.1f} Hz")

        # Buffer status
        if state.buffer_ready:
            buffer_status = "[green]Ready[/]"
        else:
            buffer_status = f"[yellow]Filling {state.buffer_fill:.0%}[/]"
        table.add_row("Buffer", buffer_status)

        return Panel(
            table,
            title=f"[bold]{self.title}[/]",
            subtitle="Press Ctrl+C to stop",
        )
