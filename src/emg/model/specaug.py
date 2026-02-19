import torch
import torch.nn as nn


class _AxesMask(nn.Module):
    """Mask a contiguous range along one or more axes."""

    def __init__(
        self,
        max_mask_length: int,
        axes: tuple[int, ...],
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        if max_mask_length < 0:
            raise ValueError("max_mask_length must be non-negative")
        if len(axes) == 0:
            raise ValueError("axes must not be empty")

        self.max_mask_length = max_mask_length
        self.axes = axes
        self.mask_value = mask_value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.max_mask_length == 0:
            return inputs

        max_length = min(self.max_mask_length, *(inputs.shape[axis] for axis in self.axes))
        if max_length <= 0:
            return inputs

        mask_length = int(
            torch.randint(max_length + 1, size=(), device=inputs.device).item()
        )
        if mask_length == 0:
            return inputs

        slices = [slice(None)] * inputs.ndim
        for axis in self.axes:
            axis_size = inputs.shape[axis]
            start = int(
                torch.randint(
                    axis_size - mask_length + 1, size=(), device=inputs.device
                ).item()
            )
            slices[axis] = slice(start, start + mask_length)

        outputs = inputs.clone()
        outputs[tuple(slices)] = self.mask_value
        return outputs


class SpecMaskAug(nn.Module):
    """SpecAugment-style masking for generic feature tensors."""

    def __init__(
        self,
        max_num_masks: list[int],
        max_mask_lengths: list[int],
        dims: str = "TF",
        axes_by_coord: dict[str, tuple[int, ...]] | None = None,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        if len(max_num_masks) != len(dims):
            raise ValueError("max_num_masks length must match dims length")
        if len(max_mask_lengths) != len(dims):
            raise ValueError("max_mask_lengths length must match dims length")
        if len(set(dims)) != len(dims):
            raise ValueError("dims must not contain duplicates")

        if axes_by_coord is None:
            axes_by_coord = {"N": (0,), "C": (1,), "F": (2,), "T": (3,)}

        self.max_num_masks = []
        self.masks = nn.ModuleList()
        for idx, dim in enumerate(dims):
            if dim not in axes_by_coord:
                raise ValueError(f"Unsupported dim '{dim}' for axes_by_coord")
            if max_num_masks[idx] < 0:
                raise ValueError("max_num_masks must be non-negative")

            self.max_num_masks.append(max_num_masks[idx])
            self.masks.append(
                _AxesMask(
                    max_mask_length=max_mask_lengths[idx],
                    axes=axes_by_coord[dim],
                    mask_value=mask_value,
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs

        outputs = inputs
        for max_masks, mask in zip(self.max_num_masks, self.masks):
            num_masks = int(
                torch.randint(max_masks + 1, size=(), device=inputs.device).item()
            )
            for _ in range(num_masks):
                outputs = mask(outputs)
        return outputs


class MPFSpecMaskAug(SpecMaskAug):
    """SpecAugment for MPF tensors shaped (batch, freq, ch, ch, time)."""

    def __init__(
        self,
        max_num_masks: list[int],
        max_mask_lengths: list[int],
        dims: str = "TF",
        mask_value: float = 0.0,
    ) -> None:
        super().__init__(
            max_num_masks=max_num_masks,
            max_mask_lengths=max_mask_lengths,
            dims=dims,
            axes_by_coord={"N": (0,), "F": (1,), "C": (2, 3), "T": (4,)},
            mask_value=mask_value,
        )


class SpectrogramSpecMaskAug(SpecMaskAug):
    """SpecAugment for spectrogram tensors shaped (batch, channels, freq, time)."""

    def __init__(
        self,
        max_num_masks: list[int],
        max_mask_lengths: list[int],
        dims: str = "TF",
        mask_value: float = 0.0,
    ) -> None:
        super().__init__(
            max_num_masks=max_num_masks,
            max_mask_lengths=max_mask_lengths,
            dims=dims,
            axes_by_coord={"N": (0,), "C": (1,), "F": (2,), "T": (3,)},
            mask_value=mask_value,
        )
