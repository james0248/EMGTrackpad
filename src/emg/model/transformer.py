import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from emg.util.mlp import mlp


class STFTFeatures(nn.Module):
    """STFT feature extraction returning raw spectrum."""

    def __init__(self, fft_window: int, fft_stride: int):
        super().__init__()
        self.fft_window = fft_window
        self.fft_stride = fft_stride
        self.num_freqs = fft_window // 2 + 1

        hann_window = torch.hann_window(fft_window, periodic=False)
        self.register_buffer("hann_window", hann_window)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = emg.shape
        emg = rearrange(emg, "b c t -> (b c) t")
        emg = F.normalize(emg, dim=-1)

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
        spectrum = rearrange(
            spectrum, "(b c) f t -> b c t f", b=batch_size, c=num_channels
        )
        return spectrum


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for time dimension."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]


class CircularChannelEncoding(nn.Module):
    """Circular sinusoidal encoding for EMG channels arranged in a ring.

    EMG channels are physically arranged in a circle around the wrist:
        Ch0
      Ch7   Ch1
    Ch6       Ch2
      Ch5   Ch3
        Ch4

    This encoding captures the circular topology using sin/cos at multiple
    frequencies of the channel angle theta_k = 2*pi*k/num_channels.
    """

    def __init__(self, d_model: int, num_channels: int, num_frequencies: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        pe = torch.zeros(num_channels, d_model)
        for k in range(num_channels):
            theta = 2 * math.pi * k / num_channels
            for f in range(num_frequencies):
                freq = f + 1
                idx = f * 2
                if idx < d_model:
                    pe[k, idx] = math.sin(freq * theta)
                if idx + 1 < d_model:
                    pe[k, idx + 1] = math.cos(freq * theta)
        self.register_buffer("pe", pe)

    def forward(self) -> torch.Tensor:
        return self.pe


class STFTTransformerBase(nn.Module):
    """Base Transformer model for STFT-based EMG processing."""

    def __init__(
        self,
        num_channels: int,
        fft_window: int,
        fft_stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.d_model = d_model
        self.dropout = dropout

        self.stft_features = STFTFeatures(fft_window=fft_window, fft_stride=fft_stride)
        self.input_projection = nn.Linear(self.stft_features.num_freqs, d_model)
        self.time_pe = SinusoidalPositionalEncoding(d_model, max_len=1000)
        self.channel_pe = CircularChannelEncoding(
            d_model, num_channels, num_frequencies=num_channels // 2
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, emg: torch.Tensor) -> torch.Tensor:
        """Encode EMG signal to CLS token representation."""
        batch_size = emg.shape[0]

        stft = self.stft_features(emg)
        num_time_frames = stft.shape[2]

        x = self.input_projection(stft)

        time_pe = self.time_pe(num_time_frames)
        channel_pe = self.channel_pe()
        x = x + time_pe.unsqueeze(0).unsqueeze(0) + channel_pe.unsqueeze(0).unsqueeze(2)

        x = rearrange(x, "b c t d -> b (t c) d")

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        return x[:, 0]


class STFTTransformerClickClassifier(STFTTransformerBase):
    """Transformer-based EMG click classifier using STFT features."""

    def __init__(
        self,
        num_channels: int,
        fft_window: int,
        fft_stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__(
            num_channels=num_channels,
            fft_window=fft_window,
            fft_stride=fft_stride,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.ln = nn.LayerNorm(d_model)
        self.classifier = mlp(
            [d_model, d_model, 3], activation=nn.GELU, dropout=dropout
        )

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_output = self.encode(emg)
        cls_output = self.ln(cls_output)
        return {"click": self.classifier(cls_output)}


class STFTTransformerController(STFTTransformerBase):
    """Transformer-based EMG controller using STFT features."""

    def __init__(
        self,
        num_channels: int,
        fft_window: int,
        fft_stride: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__(
            num_channels=num_channels,
            fft_window=fft_window,
            fft_stride=fft_stride,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.ln = nn.LayerNorm(d_model)
        self.dxdy_head = mlp([d_model, 2], activation=nn.GELU, dropout=dropout)
        self.actions_head = mlp([d_model, 3], activation=nn.GELU, dropout=dropout)

    def forward(self, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_output = self.encode(emg)
        cls_output = self.ln(cls_output)
        return {
            "dxdy": self.dxdy_head(cls_output),
            "actions": self.actions_head(cls_output),
        }
