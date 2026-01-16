import torch
import torch.nn as nn

from util.mlp import mlp


class StackedRMSFeatures(nn.Module):
    """Extracts log-scaled RMS features from non-overlapping sub-windows."""

    def __init__(self, window_length: int):
        super().__init__()
        self.window_length = window_length

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        windows = emg.unfold(2, self.window_length, self.window_length)
        rms = torch.sqrt(torch.mean(windows**2, dim=-1) + 1e-8)
        log_rms = torch.log(rms + 1e-8)
        return log_rms.flatten(start_dim=1)


class RMSMLPModel(nn.Module):
    """MLP model using stacked RMS features for EMG-based cursor control."""

    def __init__(
        self,
        num_channels: int,
        hidden_dims: list[int],
        emg_sample_rate: float,
        window_length_s: float,
        rms_window_s: float,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)
        rms_window_samples = int(rms_window_s * emg_sample_rate)
        num_windows = window_samples // rms_window_samples

        self.rms_features = StackedRMSFeatures(rms_window_samples)

        input_dim = num_channels * num_windows
        self.mlp = mlp([input_dim, *hidden_dims], nn.ReLU, last_activation=nn.ReLU)
        self.dxdy_head = nn.Linear(hidden_dims[-1], 2)
        self.actions_head = nn.Linear(hidden_dims[-1], 3)

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            emg: Raw EMG signal (batch, channels, time)

        Returns:
            Dictionary with 'dxdy' (cursor movement) and 'actions' (gesture logits)
        """
        features = self.rms_features(emg)
        hidden = self.mlp(features)
        return {
            "dxdy": self.dxdy_head(hidden),
            "actions": self.actions_head(hidden),
        }


class RMSMLPClickClassifier(nn.Module):
    """3-class click classifier: nothing (0), left (1), right (2).

    Uses stacked RMS features to capture the temporal profile of EMG signals
    for detecting discrete click events.
    """

    def __init__(
        self,
        num_channels: int,
        hidden_dims: list[int],
        emg_sample_rate: float,
        window_length_s: float,
        rms_window_s: float,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)
        rms_window_samples = int(rms_window_s * emg_sample_rate)
        num_windows = window_samples // rms_window_samples

        self.rms_features = StackedRMSFeatures(rms_window_samples)

        input_dim = num_channels * num_windows
        self.mlp = mlp(
            dims=[input_dim] + hidden_dims + [3], activation=nn.ReLU, dropout=0.1
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            emg: Raw EMG signal (batch, channels, time)

        Returns:
            Dictionary with 'click' logits (batch, 3)
        """
        features = self.rms_features(emg)
        logits = self.mlp(features)
        return {"click": logits}
