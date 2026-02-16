from dataclasses import dataclass, field

import numpy as np
import torch

from emg.inference.trackpad import VirtualTrackpad


@dataclass
class TrackpadCommand:
    """Unified command format for trackpad control."""

    dx: float = 0.0
    dy: float = 0.0
    click_state: int = 0  # 0=none, 1=left, 2=right
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0
    is_move_enabled: bool = True
    is_scroll_enabled: bool = True

    def apply_to(self, trackpad: VirtualTrackpad) -> None:
        trackpad.apply(
            self.dx,
            self.dy,
            self.click_state,
            self.scroll_dx,
            self.scroll_dy,
            self.is_move_enabled,
            self.is_scroll_enabled,
        )


class ClickSmoother:
    """
    Anti-glitch smoothing for click predictions.

    Uses confidence threshold and hysteresis to prevent stuttering:
    - Confidence threshold: reject uncertain predictions
    - Hysteresis: require state to persist for N frames before changing

    Tracks unified click state: 0=none, 1=left, 2=right (ClickState values)
    """

    def __init__(self, confidence_threshold: float = 0.7, hold_frames: int = 2):
        self.confidence_threshold = confidence_threshold
        self.hold_frames = hold_frames

        self._current_state: int = 0
        self._pending_state: int | None = None
        self._pending_count = 0

    def update(self, new_state: int, confidence: float) -> int:
        if confidence < self.confidence_threshold:
            self._pending_state = None
            self._pending_count = 0
            return self._current_state

        # Check if this is a state change
        if new_state != self._current_state:
            if self._pending_state == new_state:
                self._pending_count += 1
                if self._pending_count >= self.hold_frames:
                    self._current_state = new_state
                    self._pending_state = None
                    self._pending_count = 0
            else:
                self._pending_state = new_state
                self._pending_count = 1
        else:
            self._pending_state = None
            self._pending_count = 0

        return self._current_state


class MovementProcessor:
    """
    Cursor movement post-processing.

    Handles:
    - Denormalization from model output to pixel deltas
    - Dead zone filtering (discard small movements)
    - Sensitivity scaling
    """

    def __init__(
        self,
        dxdy_mean: np.ndarray,
        dxdy_std: np.ndarray,
        sensitivity: float = 1.0,
        dead_zone: float = 1.0,
    ):
        self.dxdy_mean = dxdy_mean
        self.dxdy_std = dxdy_std
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone

    def process(self, dxdy_normalized: np.ndarray) -> tuple[int, int]:
        dx = dxdy_normalized[0] * self.dxdy_std[0] + self.dxdy_mean[0]
        dy = dxdy_normalized[1] * self.dxdy_std[1] + self.dxdy_mean[1]

        if abs(dx) <= self.dead_zone:
            dx = 0.0
        if abs(dy) <= self.dead_zone:
            dy = 0.0

        dx = dx * self.sensitivity
        dy = dy * self.sensitivity
        return dx, dy


class ActionProcessor:
    """
    Controller action thresholding with optional smoothing.

    Handles sigmoid + threshold for left_click, right_click, scroll actions.
    """

    def __init__(
        self,
        click_threshold: float = 0.5,
        scroll_threshold: float = 0.5,
        scroll_amount: int = 10,
        click_smoother: ClickSmoother | None = None,
    ):
        self.click_threshold = click_threshold
        self.scroll_threshold = scroll_threshold
        self.scroll_amount = scroll_amount
        self.click_smoother = click_smoother

    def process(self, actions_logits: np.ndarray) -> tuple[int, bool, int]:
        probs = torch.sigmoid(torch.tensor(actions_logits))

        left_prob = float(probs[0])
        right_prob = float(probs[1])
        scroll_prob = float(probs[2])

        # Pick argmax between left and right
        if left_prob >= right_prob:
            argmax_state = 1  # ClickState.LEFT
            argmax_prob = left_prob
        else:
            argmax_state = 2  # ClickState.RIGHT
            argmax_prob = right_prob

        # Apply threshold gate
        if argmax_prob > self.click_threshold:
            predicted_state = argmax_state
            click_confidence = argmax_prob
        else:
            predicted_state = 0  # ClickState.NONE
            click_confidence = 1.0 - argmax_prob

        # Apply smoothing if available
        if self.click_smoother is not None:
            click_state = self.click_smoother.update(predicted_state, click_confidence)
        else:
            click_state = predicted_state

        scroll_active = scroll_prob > self.scroll_threshold
        scroll_dy = self.scroll_amount if scroll_active else 0

        return click_state, scroll_active, scroll_dy


@dataclass
class ClickModelInterface:
    """Interface for click classification model output."""

    confidence_threshold: float = 0.7
    hold_frames: int = 2
    _smoother: ClickSmoother = field(init=False)

    def __post_init__(self):
        self._smoother = ClickSmoother(self.confidence_threshold, self.hold_frames)

    def process(self, model_output: dict[str, torch.Tensor]) -> TrackpadCommand:
        logits = model_output["click"]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = int(logits.argmax(dim=-1).item())

        # Get confidence for the predicted class
        confidence = float(probs[pred_class])

        # Apply smoothing to unified click state (0=none, 1=left, 2=right)
        smoothed_state = self._smoother.update(pred_class, confidence)

        return TrackpadCommand(click_state=smoothed_state)

    def get_prediction_info(
        self, model_output: dict[str, torch.Tensor]
    ) -> tuple[int, np.ndarray]:
        """Get raw prediction info for display purposes."""
        logits = model_output["click"]
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = int(logits.argmax(dim=-1).item())
        return pred_class, probs


@dataclass
class ControllerModelInterface:
    """Interface for controller model output."""

    dxdy_mean: np.ndarray
    dxdy_std: np.ndarray
    sensitivity: float = 1.0
    click_threshold: float = 0.5
    click_confidence_threshold: float = 0.6
    click_hold_frames: int = 2
    scroll_threshold: float = 0.5
    scroll_amount: int = 10
    dead_zone: float = 1.0
    _movement_processor: MovementProcessor = field(init=False)
    _action_processor: ActionProcessor = field(init=False)

    def __post_init__(self):
        click_smoother = ClickSmoother(
            self.click_confidence_threshold, self.click_hold_frames
        )
        self._movement_processor = MovementProcessor(
            self.dxdy_mean, self.dxdy_std, self.sensitivity, self.dead_zone
        )
        self._action_processor = ActionProcessor(
            self.click_threshold,
            self.scroll_threshold,
            self.scroll_amount,
            click_smoother,
        )

    def process(self, model_output: dict[str, torch.Tensor]) -> TrackpadCommand:
        """
        Process controller model output to TrackpadCommand.

        Args:
            model_output: Dict with "dxdy" (1, 2) and "actions" (1, 3)

        Returns:
            TrackpadCommand with movement and action states
        """
        # Process movement
        dxdy_normalized = model_output["dxdy"].cpu().numpy()[0]
        dx, dy = self._movement_processor.process(dxdy_normalized)

        # Process actions
        actions_logits = model_output["actions"].cpu().numpy()[0]
        click_state, _, scroll_dy = self._action_processor.process(actions_logits)

        return TrackpadCommand(
            dx=float(dx),
            dy=float(dy),
            click_state=click_state,
            scroll_dy=float(scroll_dy),
        )

    def get_raw_values(
        self, model_output: dict[str, torch.Tensor]
    ) -> tuple[float, float, bool, bool, bool]:
        """Get raw denormalized values for display purposes."""
        dxdy_normalized = model_output["dxdy"].cpu().numpy()[0]
        dx = dxdy_normalized[0] * self.dxdy_std[0] + self.dxdy_mean[0]
        dy = dxdy_normalized[1] * self.dxdy_std[1] + self.dxdy_mean[1]

        actions_logits = model_output["actions"].cpu().numpy()[0]
        probs = 1.0 / (1.0 + np.exp(-actions_logits))
        left_click = probs[0] > self.click_threshold
        right_click = probs[1] > self.click_threshold
        scroll = probs[2] > self.scroll_threshold

        return float(dx), float(dy), bool(left_click), bool(right_click), bool(scroll)
