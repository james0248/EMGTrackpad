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
from torch.utils.data import DataLoader, random_split

from emg.datasets.controller_dataset import make_controller_dataset
from emg.util.device import get_device

logger = logging.getLogger(__name__)

ACTION_NAMES = ["move", "scroll", "left", "right"]


def predict_action(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Predict actions from logits using BCE inference logic.

    Args:
        logits: Raw logits of shape (batch, num_actions)
        threshold: Probability threshold for selecting an action

    Returns:
        Predicted action index. If no action exceeds threshold, returns -1 (nothing).
        If multiple actions exceed threshold, returns the one with highest probability.
    """
    probs = torch.sigmoid(logits)
    above_threshold = probs > threshold

    # Default to -1 (nothing) if no action exceeds threshold
    preds = torch.full((logits.shape[0],), -1, dtype=torch.long, device=logits.device)

    # For samples with at least one action above threshold, pick the highest prob
    any_above = above_threshold.any(dim=-1)
    preds[any_above] = probs[any_above].argmax(dim=-1)

    return preds


def compute_pos_weights(dataloader: DataLoader, num_actions: int = 4) -> torch.Tensor:
    """Compute positive class weights for BCEWithLogitsLoss.

    Returns:
        Tensor of pos_weight for each action (num_negatives / num_positives).
    """
    all_actions = torch.cat([batch["actions"] for batch in dataloader])
    pos_counts = all_actions.sum(dim=0)
    neg_counts = len(all_actions) - pos_counts

    return neg_counts / (pos_counts + 1e-8)


def save_confusion_matrix(
    cm: torch.Tensor, class_names: list[str], save_path: Path
) -> None:
    """Save confusion matrix as percentage heatmap image."""
    cm_np = cm.cpu().numpy()
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_np, row_sums, where=row_sums != 0) * 100

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
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    action_loss_fn: nn.Module,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    dxdy_loss_fn = nn.MSELoss()
    num_actions = len(ACTION_NAMES)

    all_preds = []
    all_targets = []
    total_action_loss = 0.0
    total_dxdy_loss = 0.0

    for batch in dataloader:
        emg = batch["emg"].to(device)
        action_targets = batch["actions"].to(device)
        dxdy_targets = batch["dxdy"].to(device)

        out = model(emg)
        action_logits = out["actions"]
        dxdy_pred = out["dxdy"]

        total_action_loss += action_loss_fn(action_logits, action_targets).item()
        total_dxdy_loss += dxdy_loss_fn(dxdy_pred, dxdy_targets).item()

        # Predict using threshold logic
        preds = predict_action(action_logits)
        # Convert target actions to class index (-1 for nothing)
        target_idx = torch.where(
            action_targets.any(dim=-1),
            action_targets.argmax(dim=-1),
            torch.tensor(-1, device=device),
        )
        all_preds.append(preds)
        all_targets.append(target_idx)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Include "nothing" as class index -1 -> shift to 0 for metrics
    # Classes: 0=nothing, 1=left, 2=right, 3=scroll
    preds_shifted = preds + 1
    targets_shifted = targets + 1
    num_classes = num_actions + 1
    class_names = ["nothing"] + ACTION_NAMES

    metrics: dict[str, float] = {
        "action_loss": total_action_loss / len(dataloader),
        "dxdy_loss": total_dxdy_loss / len(dataloader),
        "accuracy": (preds == targets).float().mean().item(),
    }

    # Compute per-class metrics using one-hot encoding
    preds_onehot = torch.nn.functional.one_hot(preds_shifted, num_classes).bool()
    targets_onehot = torch.nn.functional.one_hot(targets_shifted, num_classes).bool()

    tp = (preds_onehot & targets_onehot).sum(dim=0).float()
    fp = (preds_onehot & ~targets_onehot).sum(dim=0).float()
    fn = (~preds_onehot & targets_onehot).sum(dim=0).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics["f1_macro"] = f1.mean().item()

    for c, name in enumerate(class_names):
        metrics[f"precision_{name}"] = precision[c].item()
        metrics[f"recall_{name}"] = recall[c].item()
        metrics[f"f1_{name}"] = f1[c].item()

    # Compute confusion matrix (vectorized)
    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=preds.device
    )
    indices = targets_shifted * num_classes + preds_shifted
    confusion.view(-1).scatter_add_(
        0, indices, torch.ones_like(indices, dtype=torch.long)
    )
    metrics["confusion_matrix"] = confusion
    metrics["class_names"] = class_names

    return metrics


@hydra.main(version_base=None, config_path="config/controller")
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

    dataset = make_controller_dataset(
        session_files,
        window_length_s=cfg.training.window_length_s,
        highpass_freq=cfg.preprocessing.highpass_freq,
        emg_scale=cfg.preprocessing.emg_scale,
        stride_s=cfg.training.stride_s,
        jitter=cfg.training.get("jitter", False),
    )

    # Log normalization statistics
    logger.info(f"dxdy normalization - mean: {dataset.dxdy_mean.tolist()}")
    logger.info(f"dxdy normalization - std: {dataset.dxdy_std.tolist()}")

    # Split into train and eval
    eval_split = cfg.training.eval_split
    train_dataset, eval_dataset = random_split(dataset, [1 - eval_split, eval_split])
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

    # Loss functions
    num_actions = len(ACTION_NAMES)
    dxdy_loss_fn = nn.MSELoss()

    # BCE loss with optional pos_weight for class imbalance
    if cfg.training.use_class_weights:
        pos_weights = compute_pos_weights(train_loader, num_actions)
        pos_weights = pos_weights.to(device)
        logger.info(f"Pos weights: {pos_weights.tolist()}")
        action_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        logger.info("Using unweighted BCE loss")
        action_loss_fn = nn.BCEWithLogitsLoss()

    # Loss weights
    action_weight = cfg.training.get("action_loss_weight", 1.0)
    dxdy_weight = cfg.training.get("dxdy_loss_weight", 1.0)

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_dxdy_loss = 0.0

        for batch in train_loader:
            emg = batch["emg"].to(device)
            action_targets = batch["actions"].to(device)
            dxdy_targets = batch["dxdy"].to(device)

            optimizer.zero_grad()
            out = model(emg)
            action_logits = out["actions"]
            dxdy_pred = out["dxdy"]

            action_loss = action_loss_fn(action_logits, action_targets)
            dxdy_loss = dxdy_loss_fn(dxdy_pred, dxdy_targets)
            loss = action_weight * action_loss + dxdy_weight * dxdy_loss

            loss.backward()

            if cfg.training.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.training.grad_clip_norm
                )
            optimizer.step()

            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_dxdy_loss += dxdy_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_action_loss = total_action_loss / len(train_loader)
        avg_dxdy_loss = total_dxdy_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch + 1:3d}/{cfg.training.num_epochs} | "
            f"Loss: {avg_loss:.4f} (action: {avg_action_loss:.4f}, dxdy: {avg_dxdy_loss:.4f})"
        )

        # Evaluation
        if (epoch + 1) % cfg.training.eval_interval == 0:
            metrics = evaluate(model, eval_loader, device, action_loss_fn)
            logger.info(
                f"[Eval] Action Loss: {metrics['action_loss']:.4f} | "
                f"dxdy Loss: {metrics['dxdy_loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1 (macro): {metrics['f1_macro']:.4f}"
            )
            logger.info(
                f"       F1 nothing: {metrics['f1_nothing']:.4f} | "
                f"F1 move: {metrics['f1_move']:.4f} | "
                f"F1 scroll: {metrics['f1_scroll']:.4f} | "
                f"F1 left: {metrics['f1_left']:.4f} | "
                f"F1 right: {metrics['f1_right']:.4f}"
            )
            save_confusion_matrix(
                metrics["confusion_matrix"],
                metrics["class_names"],
                plots_dir / f"confusion_epoch_{epoch + 1:04d}.png",
            )

            model.train()

        # Save checkpoint with normalization stats
        if (epoch + 1) % cfg.training.save_interval == 0:
            save_path = checkpoint_dir / f"model_epoch_{epoch + 1:04d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dxdy_mean": dataset.dxdy_mean,
                    "dxdy_std": dataset.dxdy_std,
                },
                save_path,
            )
            logger.info(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    train()
