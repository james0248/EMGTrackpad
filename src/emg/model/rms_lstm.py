import torch
import torch.nn as nn
from einops import rearrange


class FrequencyRMSFeaturesSequence(nn.Module):
    """Extracts frequency-domain RMS features as a sequence for LSTM input."""

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
        self.fft_stride = fft_stride
        self.frequency_bins = frequency_bins
        self.register_buffer("freq_masks", self._build_frequency_masks())

        hann_window = torch.hann_window(fft_window, periodic=False)
        self.register_buffer("hann_window", hann_window)
        self.window_normalization_factor = torch.linalg.vector_norm(hann_window)

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
            rms_by_bin, "(b c) n t -> b t (c n)", b=batch_size, c=num_channels
        )
        return rms_by_bin

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


class FrequencyRMSLSTMModel(nn.Module):
    """LSTM model using frequency-domain RMS features for EMG-based cursor control."""

    def __init__(
        self,
        num_channels: int,
        emg_sample_rate: float,
        window_length_s: float,
        fft_window: int,
        fft_stride: int,
        frequency_bins: list[tuple[float, float]],
        projection_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        num_bins = len(frequency_bins)
        input_dim = num_channels * num_bins

        self.freq_rms_features = FrequencyRMSFeaturesSequence(
            sample_rate=emg_sample_rate,
            fft_window=fft_window,
            fft_stride=fft_stride,
            frequency_bins=frequency_bins,
        )

        self.projection = nn.Linear(input_dim, projection_dim)

        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.dxdy_head = nn.Linear(lstm_hidden_dim, 4)
        self.actions_head = nn.Linear(lstm_hidden_dim, 4)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            emg: Raw EMG signal (batch, channels, time)

        Returns:
            Dictionary with 'dxdy' (cursor movement) and 'actions' (gesture logits)
        """
        features = self.freq_rms_features(emg)
        features = self.projection(features)
        _, (hidden, _) = self.lstm(features)
        last_hidden = hidden[-1]

        return {
            "dxdy": self.dxdy_head(last_hidden),
            "actions": self.actions_head(last_hidden),
        }
