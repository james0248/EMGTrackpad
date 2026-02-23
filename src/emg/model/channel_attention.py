import math
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange

from emg.util.mlp import mlp

ChannelPoolingMode = Literal["mean", "cls"]


class STFTFeatures(nn.Module):
    """STFT feature extraction returning log-scaled spectrum."""

    def __init__(
        self,
        fft_window: int,
        fft_stride: int,
        spec_augment: nn.Module | None = None,
    ):
        super().__init__()
        self.fft_window = fft_window
        self.fft_stride = fft_stride
        self.num_freqs = fft_window // 2 + 1
        self.spec_augment = spec_augment

        hann_window = torch.hann_window(fft_window, periodic=False)
        self.register_buffer("hann_window", hann_window)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = emg.shape
        emg = rearrange(emg, "b c t -> (b c) t")

        spectrum = torch.stft(
            emg,
            n_fft=self.fft_window,
            hop_length=self.fft_stride,
            window=self.hann_window,
            center=False,
            onesided=True,
            return_complex=True,
        ).abs()

        spectrum = torch.log1p(spectrum)
        if self.spec_augment is not None:
            spectrum = self.spec_augment(spectrum.unsqueeze(1)).squeeze(1)
        return rearrange(spectrum, "(b c) f t -> b c t f", b=batch_size, c=num_channels)


class CircularChannelEncoding(nn.Module):
    """Circular sinusoidal encoding for channels arranged in a ring."""

    def __init__(self, d_model: int, num_channels: int, num_frequencies: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        pe = torch.zeros(num_channels, d_model)
        for k in range(num_channels):
            theta = 2 * math.pi * k / num_channels
            for freq_idx in range(num_frequencies):
                freq = freq_idx + 1
                dim = 2 * freq_idx
                if dim < d_model:
                    pe[k, dim] = math.sin(freq * theta)
                if dim + 1 < d_model:
                    pe[k, dim + 1] = math.cos(freq * theta)
        self.register_buffer("pe", pe)

    def forward(self) -> torch.Tensor:
        return self.pe


class _ChannelAttentionEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer for channel-wise attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = mlp(
            [d_model, dim_feedforward, d_model], activation=nn.GELU, dropout=dropout
        )
        self.ffn_residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = x + self.attn_dropout(attn_out)

        ffn_out = self.ffn(x)
        x = x + self.ffn_residual_dropout(ffn_out)
        return x


class _ChannelAttentionEncoder(nn.Module):
    """Stacked transformer encoder for channel token processing."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ChannelAttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _compute_num_time_frames(
    window_length_s: float,
    emg_sample_rate: float,
    fft_window: int,
    fft_stride: int,
) -> int:
    window_samples = int(window_length_s * emg_sample_rate)
    if window_samples < fft_window:
        raise ValueError(
            "window_length_s * emg_sample_rate must be >= fft_window for STFT."
        )

    num_time_frames = (window_samples - fft_window) // fft_stride + 1
    if num_time_frames <= 0:
        raise ValueError(
            "STFT configuration produces no time frames. "
            "Increase window length or reduce fft_window/fft_stride."
        )
    return num_time_frames


class ChannelAttentionFeatures(nn.Module):
    """STFT features + channel transformer with configurable channel pooling."""

    def __init__(
        self,
        num_channels: int,
        emg_sample_rate: float,
        window_length_s: float,
        fft_window: int,
        fft_stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: ChannelPoolingMode = "mean",
        spec_augment: nn.Module | None = None,
    ):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(
                f"Invalid pooling mode: {pooling}. Expected one of: mean, cls."
            )

        self.num_channels = num_channels
        self.emg_sample_rate = emg_sample_rate
        self.window_length_s = window_length_s
        self.d_model = d_model
        self.pooling = pooling
        self.num_time_frames = _compute_num_time_frames(
            window_length_s=window_length_s,
            emg_sample_rate=emg_sample_rate,
            fft_window=fft_window,
            fft_stride=fft_stride,
        )

        self.stft_features = STFTFeatures(
            fft_window=fft_window,
            fft_stride=fft_stride,
            spec_augment=spec_augment,
        )
        self.input_projection = nn.Linear(self.stft_features.num_freqs, d_model)
        self.channel_pe = CircularChannelEncoding(
            d_model=d_model,
            num_channels=num_channels,
            num_frequencies=num_channels // 2,
        )
        self.transformer = _ChannelAttentionEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        """Return pooled features of shape (batch, time, d_model)."""
        batch_size = emg.shape[0]

        stft = self.stft_features(emg)  # (b, c, t, f)
        num_time_frames = stft.shape[2]

        x = self.input_projection(stft)  # (b, c, t, d)
        x = x + self.channel_pe().unsqueeze(0).unsqueeze(2)

        # Channel-wise attention per time frame.
        x = rearrange(x, "b c t d -> (b t) c d")
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)

        if self.pooling == "cls":
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        x = rearrange(x, "(b t) d -> b t d", b=batch_size, t=num_time_frames)
        return x


class ChannelAttentionMLPController(nn.Module):
    """Controller with ChannelAttention features followed by an MLP trunk."""

    def __init__(
        self,
        num_channels: int,
        emg_sample_rate: float,
        window_length_s: float,
        fft_window: int,
        fft_stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        pooling: ChannelPoolingMode = "mean",
        spec_augment: nn.Module | None = None,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden dimension.")

        self.features = ChannelAttentionFeatures(
            num_channels=num_channels,
            emg_sample_rate=emg_sample_rate,
            window_length_s=window_length_s,
            fft_window=fft_window,
            fft_stride=fft_stride,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
            spec_augment=spec_augment,
        )

        input_dim = self.features.num_time_frames * d_model
        self.mlp = mlp(
            dims=[input_dim, *hidden_dims],
            activation=nn.GELU,
            last_activation=nn.GELU,
            dropout=dropout,
            norm=nn.LayerNorm,
        )
        self.dxdy_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 4],
            activation=nn.GELU,
            dropout=dropout,
        )
        self.actions_head = mlp(
            dims=[hidden_dims[-1], hidden_dims[-1], 4],
            activation=nn.GELU,
            dropout=dropout,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(emg)  # (b, t, d)
        features = features.flatten(start_dim=1)
        hidden = self.mlp(features)
        return {
            "dxdy": self.dxdy_head(hidden),
            "actions": self.actions_head(hidden),
        }


class ChannelAttentionLSTMController(nn.Module):
    """Controller with ChannelAttention features followed by an LSTM head."""

    def __init__(
        self,
        num_channels: int,
        emg_sample_rate: float,
        window_length_s: float,
        fft_window: int,
        fft_stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        projection_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        dropout: float = 0.1,
        pooling: ChannelPoolingMode = "mean",
        spec_augment: nn.Module | None = None,
    ):
        super().__init__()

        self.features = ChannelAttentionFeatures(
            num_channels=num_channels,
            emg_sample_rate=emg_sample_rate,
            window_length_s=window_length_s,
            fft_window=fft_window,
            fft_stride=fft_stride,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pooling=pooling,
            spec_augment=spec_augment,
        )
        self.projection = nn.Linear(d_model, projection_dim)
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
        features = self.features(emg)  # (b, t, d)
        features = self.projection(features)
        _, (hidden, _) = self.lstm(features)
        last_hidden = hidden[-1]
        return {
            "dxdy": self.dxdy_head(last_hidden),
            "actions": self.actions_head(last_hidden),
        }
