import logging
import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from emg.datasets.click_dataset import make_click_dataset
from emg.util.device import get_device

logger = logging.getLogger(__name__)

CLASS_NAMES = ["nothing", "left", "right"]


def compute_class_weights(dataloader: DataLoader, num_classes: int = 3) -> torch.Tensor:
    """Compute class weights for handling class imbalance.

    Returns:
        Tensor of weights for each class, inversely proportional to frequency.
    """
    all_targets = torch.cat([batch["click"] for batch in dataloader])
    counts = torch.bincount(all_targets, minlength=num_classes).float()
    total = counts.sum()
    return total / (num_classes * counts + 1e-8)


def save_confusion_matrix(
    cm: torch.Tensor, class_names: list[str], save_path: Path
) -> None:
    """Save confusion matrix as percentage heatmap image."""
    cm_np = cm.cpu().numpy()
    cm_pct = cm_np / cm_np.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center", color=color)

    plt.colorbar(im, ax=ax, label="Percentage")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    num_classes = len(CLASS_NAMES)

    all_preds = []
    all_targets = []
    total_loss = 0.0

    for batch in dataloader:
        emg = batch["emg"].to(device)
        targets = batch["click"].to(device)
        logits = model(emg)["click"]

        total_loss += loss_fn(logits, targets).item()
        all_preds.append(logits.argmax(dim=-1))
        all_targets.append(targets)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    metrics: dict[str, float] = {
        "loss": total_loss / len(dataloader),
        "accuracy": (preds == targets).float().mean().item(),
    }

    # Compute per-class metrics using one-hot encoding
    preds_onehot = torch.nn.functional.one_hot(preds, num_classes).bool()
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes).bool()

    tp = (preds_onehot & targets_onehot).sum(dim=0).float()
    fp = (preds_onehot & ~targets_onehot).sum(dim=0).float()
    fn = (~preds_onehot & targets_onehot).sum(dim=0).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics["f1_macro"] = f1.mean().item()

    for c, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = precision[c].item()
        metrics[f"recall_{name}"] = recall[c].item()
        metrics[f"f1_{name}"] = f1[c].item()

    # Compute confusion matrix (vectorized)
    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=preds.device
    )
    indices = targets * num_classes + preds
    confusion.view(-1).scatter_add_(
        0, indices, torch.ones_like(indices, dtype=torch.long)
    )
    metrics["confusion_matrix"] = confusion

    return metrics


@hydra.main(version_base=None, config_path="config/click")
def train(cfg: DictConfig):
    # Set random seed for reproducibility
    if cfg.training.seed is not None:
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.training.seed)

    data_dir = Path(cfg.data_dir)
    session_files = list(data_dir.glob("session_*.h5"))
    if not session_files:
        raise ValueError(f"No session files found in {data_dir}")
    logger.info(f"Found {len(session_files)} session files")

    dataset = make_click_dataset(
        session_files,
        window_length_s=cfg.training.window_length_s,
        highpass_freq=cfg.preprocessing.highpass_freq,
        emg_scale=cfg.preprocessing.emg_scale,
        stride_s=cfg.training.stride_s,
    )

    # Split into train and eval (time-based: front=train, back=eval)
    eval_split = cfg.training.eval_split
    n = len(dataset)
    split = int(n * (1 - eval_split))
    train_dataset = Subset(dataset, range(split))
    eval_dataset = Subset(dataset, range(split, n))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(
        f"Dataset size: {len(dataset)} (train: {len(train_dataset)}, eval: {len(eval_dataset)})"
    )

    # Initialize model
    model = instantiate(cfg.model)
    device = get_device(cfg.training.device)
    model.to(device)

    # Setup output directories
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Loss function with optional class weights
    if cfg.training.use_class_weights:
        class_weights = compute_class_weights(train_loader)
        class_weights = class_weights.to(device)
        logger.info(f"Class weights: {class_weights.tolist()}")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        logger.info("Using unweighted cross-entropy loss")
        loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            emg = batch["emg"].to(device)
            targets = batch["click"].to(device)

            optimizer.zero_grad()
            out = model(emg)
            logits = out["click"]

            loss = loss_fn(logits, targets)
            loss.backward()

            if cfg.training.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.training.grad_clip_norm
                )
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch + 1:3d}/{cfg.training.num_epochs} | Loss: {avg_loss:.4f}"
        )

        # Evaluation
        if (epoch + 1) % cfg.training.eval_interval == 0:
            metrics = evaluate(model, eval_loader, device)
            logger.info(
                f"[Eval] Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1 (macro): {metrics['f1_macro']:.4f}"
            )
            logger.info(
                f"       F1 nothing: {metrics['f1_nothing']:.4f} | "
                f"F1 left: {metrics['f1_left']:.4f} | "
                f"F1 right: {metrics['f1_right']:.4f}"
            )
            save_confusion_matrix(
                metrics["confusion_matrix"],
                CLASS_NAMES,
                plots_dir / f"confusion_epoch_{epoch + 1:04d}.png",
            )

            model.train()

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            save_path = checkpoint_dir / f"model_epoch_{epoch + 1:04d}.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    train()
