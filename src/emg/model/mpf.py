import torch
import torch.nn as nn
from einops import rearrange

from emg.util.mlp import mlp


class MultivariatePowerFrequencyFeatures(nn.Module):
    """Convert raw EMG to multivariate power frequency (MPF) features."""

    def __init__(
        self,
        window_length: int,
        stride: int,
        n_fft: int,
        fft_stride: int,
        fs: float = 2000.0,
        frequency_bins: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        if window_length < n_fft:
            raise ValueError("window_length must be greater than n_fft")
        if fft_stride > n_fft:
            raise ValueError("fft_stride must be lower than or equal to n_fft")
        if fft_stride > stride:
            raise ValueError("stride must be greater than or equal to fft_stride")
        if stride % fft_stride != 0:
            raise ValueError("stride must be a multiple of fft_stride")
        if window_length % fft_stride != 0:
            raise ValueError("window_length must be a multiple of fft_stride")

        self.window_length = window_length
        self.stride = stride
        self.n_fft = n_fft
        self.fft_stride = fft_stride
        self.fs = fs
        self.frequency_bins = frequency_bins

        window = torch.hann_window(self.n_fft, periodic=False)
        self.register_buffer("window", window)
        self.window_normalization_factor = torch.linalg.vector_norm(self.window)

        if self.frequency_bins is not None:
            freq_masks = self._build_freq_masks(
                self.n_fft,
                self.fs,
                self.frequency_bins,
            )
            self.register_buffer("freq_masks", freq_masks)

        self.left_context = self.window_length - self.fft_stride + self.n_fft - 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = inputs.shape
        num_freqs = self.n_fft // 2 + 1

        outputs = (
            torch.stft(
                inputs.reshape(batch_size * num_channels, -1),
                n_fft=self.n_fft,
                hop_length=self.fft_stride,
                window=self.window,
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            / self.window_normalization_factor
        )

        outputs = outputs.unfold(
            dimension=-1,
            size=self.window_length // self.fft_stride,
            step=self.stride // self.fft_stride,
        )

        _, _, num_windows, window_size = outputs.shape
        outputs = outputs.reshape(
            batch_size,
            num_channels,
            num_freqs,
            num_windows,
            window_size,
        )

        outputs = outputs.transpose(1, 3)
        outputs = self._compute_strided_cross_spectral_density(outputs)

        if self.frequency_bins is not None:
            outputs = torch.stack(
                [
                    (outputs * freq_mask).sum(2) / freq_mask.sum(2).clamp_min(1.0)
                    for freq_mask in self.freq_masks.unbind(2)
                ],
                dim=2,
            )

        outputs = 0.5 * (outputs + outputs.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(outputs)
        eigvals = eigvals.log().nan_to_num(nan=0.0, neginf=0.0)
        outputs = (eigvecs * eigvals.unsqueeze(dim=-2)) @ eigvecs.transpose(-1, -2)

        outputs = outputs.permute(0, 2, 3, 4, 1)
        return outputs

    @staticmethod
    def _build_freq_masks(
        n_fft: int,
        fs: float,
        frequency_bins: list[tuple[float, float]],
    ) -> torch.Tensor:
        freqs_hz = torch.fft.fftfreq(n_fft, d=1.0 / fs)[: (n_fft // 2 + 1)].abs()

        freq_masks = []
        for idx, (start_freq, end_freq) in enumerate(frequency_bins):
            mask = torch.logical_and(freqs_hz > start_freq, freqs_hz <= end_freq)
            if not mask.any():
                raise ValueError(
                    f"frequency bin {idx} ({start_freq}, {end_freq}) has no FFT bins"
                )
            freq_masks.append(mask)

        stacked_masks = torch.stack(freq_masks, dim=0).to(dtype=torch.float32)
        return stacked_masks.reshape(1, 1, len(frequency_bins), len(freqs_hz), 1, 1)

    @staticmethod
    def _compute_strided_cross_spectral_density(inputs: torch.Tensor) -> torch.Tensor:
        """Compute cross-spectral density over channel pairs in each strided window."""

        input_dims = inputs.shape
        num_channels, window_size = input_dims[-2:]

        outputs = inputs.reshape(-1, num_channels, window_size)
        outputs = (outputs @ outputs.transpose(-2, -1).conj()) / window_size
        outputs = outputs.abs().pow(2)
        outputs = outputs.reshape(*input_dims[:-2], num_channels, num_channels)

        return outputs

    def compute_time_downsampling(self, emg_lengths: torch.Tensor) -> torch.Tensor:
        cospectrum_len = 1 + (emg_lengths - self.n_fft) // self.fft_stride
        return (cospectrum_len - self.window_length // self.fft_stride) // (
            self.stride // self.fft_stride
        ) + 1


class MPFMLPModel(nn.Module):
    """MLP controller using MPF features."""

    def __init__(
        self,
        num_channels: int,
        hidden_dims: list[int],
        emg_sample_rate: float,
        window_length_s: float,
        mpf_window_length: int,
        mpf_stride: int,
        n_fft: int,
        fft_stride: int,
        frequency_bins: list[tuple[float, float]] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)

        self.mpf_features = MultivariatePowerFrequencyFeatures(
            window_length=mpf_window_length,
            stride=mpf_stride,
            n_fft=n_fft,
            fft_stride=fft_stride,
            fs=emg_sample_rate,
            frequency_bins=frequency_bins,
        )

        num_windows = int(
            self.mpf_features.compute_time_downsampling(torch.tensor([window_samples]))
            .item()
        )
        if num_windows <= 0:
            raise ValueError(
                "MPF configuration produces no time windows. "
                "Increase input window length or reduce MPF window/stride."
            )

        num_freqs = len(frequency_bins) if frequency_bins is not None else (n_fft // 2 + 1)
        input_dim = num_windows * num_freqs * num_channels * num_channels

        self.mlp = mlp(
            dims=[input_dim, *hidden_dims],
            activation=nn.ReLU,
            last_activation=nn.ReLU,
            dropout=dropout,
            norm=nn.LayerNorm,
        )
        self.dxdy_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 4],
            activation=nn.ReLU,
        )
        self.actions_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 4],
            activation=nn.ReLU,
        )

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.mpf_features(emg)
        features = features.flatten(start_dim=1)
        hidden = self.mlp(features)

        return {
            "dxdy": self.dxdy_head(hidden),
            "actions": self.actions_head(hidden),
        }


class MPFLSTMModel(nn.Module):
    """LSTM controller using MPF feature sequences."""

    def __init__(
        self,
        num_channels: int,
        emg_sample_rate: float,
        window_length_s: float,
        mpf_window_length: int,
        mpf_stride: int,
        n_fft: int,
        fft_stride: int,
        frequency_bins: list[tuple[float, float]] | None,
        projection_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        window_samples = int(window_length_s * emg_sample_rate)

        self.mpf_features = MultivariatePowerFrequencyFeatures(
            window_length=mpf_window_length,
            stride=mpf_stride,
            n_fft=n_fft,
            fft_stride=fft_stride,
            fs=emg_sample_rate,
            frequency_bins=frequency_bins,
        )

        num_windows = int(
            self.mpf_features.compute_time_downsampling(torch.tensor([window_samples]))
            .item()
        )
        if num_windows <= 0:
            raise ValueError(
                "MPF configuration produces no time windows. "
                "Increase input window length or reduce MPF window/stride."
            )

        num_freqs = len(frequency_bins) if frequency_bins is not None else (n_fft // 2 + 1)
        input_dim = num_freqs * num_channels * num_channels

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
        features = self.mpf_features(emg)
        features = rearrange(features, "b f c1 c2 t -> b t (f c1 c2)")

        features = self.projection(features)
        _, (hidden, _) = self.lstm(features)
        last_hidden = hidden[-1]

        return {
            "dxdy": self.dxdy_head(last_hidden),
            "actions": self.actions_head(last_hidden),
        }
