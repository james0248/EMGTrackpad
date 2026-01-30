import torch
import torch.nn as nn
from einops import rearrange
from util.mlp import mlp


class RMSFeatures(nn.Module):
    """Extracts log-scaled RMS features from non-overlapping sub-windows."""

    def __init__(self, window_length: int):
        super().__init__()
        self.window_length = window_length

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        windows = emg.unfold(2, self.window_length, self.window_length)
        rms = torch.sqrt(torch.mean(windows**2, dim=-1))
        return rms.flatten(start_dim=1)


class FrequencyRMSFeatures(nn.Module):
    """Extracts log-scaled RMS features by frequency."""

    def __init__(
        self,
        sample_rate: float,
        fft_window: int,
        fft_stride: int,
        frequency_bins: list[tuple[float, float]],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_window = fft_window
        self.frequency_bins = frequency_bins
        self.fft_stride = fft_stride
        self.register_buffer("freq_masks", self._build_frequency_masks())

        hann_window = torch.hann_window(fft_window, periodic=False)
        self.register_buffer("hann_window", hann_window)
        self.window_normalization_factor = torch.linalg.vector_norm(self.hann_window)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = emg.shape

        emg = rearrange(emg, "b c t -> (b c) t")
        power = (
            torch.stft(
                emg,
                n_fft=self.fft_window,
                hop_length=self.fft_stride,
                window=self.hann_window,
                center=False,
                onesided=True,
                return_complex=True,
            )
            / self.window_normalization_factor
        ).abs() ** 2  # (batch * channels, freq, time)
        power = rearrange(power, "bc f t -> bc () f t")

        masked_power = power * self.freq_masks
        rms_by_bin = torch.sqrt(
            torch.sum(masked_power, dim=2) / torch.sum(self.freq_masks, dim=2)
        )
        rms_by_bin = rearrange(
            rms_by_bin, "(b c) n t -> b c t n", b=batch_size, c=num_channels
        )
        return rms_by_bin.flatten(start_dim=1)

    def _build_frequency_masks(self) -> torch.Tensor:
        num_freqs = self.fft_window // 2 + 1
        freqs = torch.fft.fftfreq(self.fft_window, d=1 / self.sample_rate)[
            :num_freqs
        ].abs()

        freq_masks = torch.stack(
            [
                torch.logical_and(freqs > min_freq, freqs <= max_freq)
                for min_freq, max_freq in self.frequency_bins
            ],
            dim=0,
        ).to(dtype=torch.float32)  # (num_bins, num_freqs)
        freq_masks = rearrange(freq_masks, "b f -> () b f ()")

        return freq_masks

    def get_rms_by_bin(self, stft: torch.Tensor) -> torch.Tensor:
        """Get RMS features by frequency bin."""
        return torch.sqrt(torch.mean(stft**2, dim=-1) + 1e-8)


class RMSMLPModel(nn.Module):
    """MLP model using stacked RMS features for EMG-based cursor control."""

    def __init__(
        self,
        num_channels: int,
        hidden_dims: list[int],
        emg_sample_rate: float,
        window_length_s: float,
        rms_window_s: float,
        dropout: float = 0.0,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)
        rms_window_samples = int(rms_window_s * emg_sample_rate)
        num_windows = window_samples // rms_window_samples

        self.rms_features = RMSFeatures(rms_window_samples)

        input_dim = num_channels * num_windows
        self.mlp = mlp(
            dims=[input_dim, *hidden_dims],
            activation=nn.ReLU,
            last_activation=nn.ReLU,
            dropout=dropout,
            norm=nn.LayerNorm,
        )
        self.dxdy_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 2],
            activation=nn.ReLU,
        )
        self.actions_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 3],
            activation=nn.ReLU,
        )

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
        dropout: float,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)
        rms_window_samples = int(rms_window_s * emg_sample_rate)
        num_windows = window_samples // rms_window_samples

        self.rms_features = RMSFeatures(rms_window_samples)

        input_dim = num_channels * num_windows
        self.mlp = mlp(
            dims=[input_dim] + hidden_dims + [3],
            activation=nn.ReLU,
            dropout=dropout,
            norm=nn.LayerNorm,
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


class FrequencyRMSMLPModel(nn.Module):
    """MLP model using frequency-domain RMS features for EMG-based cursor control."""

    def __init__(
        self,
        num_channels: int,
        hidden_dims: list[int],
        emg_sample_rate: float,
        window_length_s: float,
        fft_window: int,
        fft_stride: int,
        frequency_bins: list[tuple[float, float]],
        dropout: float = 0.0,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)
        num_time_frames = (window_samples - fft_window) // fft_stride + 1
        num_bins = len(frequency_bins)

        self.freq_rms_features = FrequencyRMSFeatures(
            sample_rate=emg_sample_rate,
            fft_window=fft_window,
            fft_stride=fft_stride,
            frequency_bins=frequency_bins,
        )

        input_dim = num_channels * num_time_frames * num_bins
        self.mlp = mlp(
            dims=[input_dim, *hidden_dims],
            activation=nn.ReLU,
            last_activation=nn.ReLU,
            dropout=dropout,
            norm=nn.LayerNorm,
        )
        self.dxdy_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 2],
            activation=nn.ReLU,
        )
        self.actions_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 3],
            activation=nn.ReLU,
        )

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            emg: Raw EMG signal (batch, channels, time)

        Returns:
            Dictionary with 'dxdy' (cursor movement) and 'actions' (gesture logits)
        """
        features = self.freq_rms_features(emg)
        hidden = self.mlp(features)
        return {
            "dxdy": self.dxdy_head(hidden),
            "actions": self.actions_head(hidden),
        }


class FrequencyRMSMLPClickClassifier(nn.Module):
    """3-class click classifier using frequency-domain RMS features.

    Uses STFT-based frequency band features to capture both temporal and spectral
    characteristics of EMG signals for detecting discrete click events.
    """

    def __init__(
        self,
        num_channels: int,
        hidden_dims: list[int],
        emg_sample_rate: float,
        window_length_s: float,
        fft_window: int,
        fft_stride: int,
        frequency_bins: list[tuple[float, float]],
        dropout: float,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)
        num_time_frames = (window_samples - fft_window) // fft_stride + 1
        num_bins = len(frequency_bins)

        self.freq_rms_features = FrequencyRMSFeatures(
            sample_rate=emg_sample_rate,
            fft_window=fft_window,
            fft_stride=fft_stride,
            frequency_bins=frequency_bins,
        )

        input_dim = num_channels * num_time_frames * num_bins
        self.mlp = mlp(
            dims=[input_dim] + hidden_dims + [3],
            activation=nn.ReLU,
            dropout=dropout,
            norm=nn.LayerNorm,
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
        features = self.freq_rms_features(emg)
        logits = self.mlp(features)
        return {"click": logits}
