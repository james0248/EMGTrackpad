import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmStartLR(LRScheduler):
    """Linear warmup followed by cosine decay.

    This scheduler is designed for epoch-level stepping and works with calling
    ``scheduler.step()`` once at the end of each epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        eta_min: float = 1e-6,
        warmup_start_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if warmup_epochs >= total_epochs:
            raise ValueError(
                f"warmup_epochs must be < total_epochs, got {warmup_epochs} >= {total_epochs}"
            )
        if not 0.0 <= warmup_start_factor <= 1.0:
            raise ValueError(
                "warmup_start_factor must be in [0, 1], "
                f"got {warmup_start_factor}"
            )

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        epoch = min(self.last_epoch, self.total_epochs - 1)

        # Warmup: start from base_lr * warmup_start_factor and linearly increase
        # to base_lr over warmup_epochs.
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            if self.warmup_epochs == 1:
                factor = 1.0
            else:
                progress = epoch / (self.warmup_epochs - 1)
                factor = self.warmup_start_factor + progress * (
                    1.0 - self.warmup_start_factor
                )
            return [base_lr * factor for base_lr in self.base_lrs]

        # Cosine decay to eta_min for remaining epochs.
        decay_epochs = self.total_epochs - self.warmup_epochs
        decay_epoch = epoch - self.warmup_epochs
        if decay_epochs == 1:
            progress = 1.0
        else:
            progress = decay_epoch / (decay_epochs - 1)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]
