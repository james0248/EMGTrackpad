import torch


class ChannelRotation:
    """Rotate EMG channels to augment data from circular armband."""

    max_offset: int = 1

    def __call__(self, emg: torch.Tensor) -> torch.Tensor:
        k = torch.randint(-self.max_offset, self.max_offset + 1, (1,)).item()
        return torch.roll(emg, shifts=k, dims=0)
