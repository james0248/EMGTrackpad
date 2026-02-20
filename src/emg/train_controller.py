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
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Subset

from emg.datasets.controller_dataset import make_controller_dataset
from emg.datasets.split import split_tail
from emg.util.device import get_device

logger = logging.getLogger(__name__)

ACTION_NAMES = ["move", "scroll", "left", "right"]
MOVEMENT_CLASS_NAMES = ["nothing", "move", "scroll"]
CLICK_CLASS_NAMES = ["nothing", "left", "right"]


def total_data_duration_s(dataset) -> float:
    """Compute total duration in seconds across all loaded sessions."""
    total = 0.0
    for session_dataset in dataset.datasets:
        if len(session_dataset.emg_timestamps) < 2:
            continue
        total += float(
            session_dataset.emg_timestamps[-1] - session_dataset.emg_timestamps[0]
        )
    return total


def predict_group_action(
    group_logits: torch.Tensor,
    thresholds: tuple[float, float],
) -> torch.Tensor:
    """Predict group action with thresholding + max-logit winner.

    Group output labels are:
        0 = nothing, 1 = first action, 2 = second action
    """
    probs = torch.sigmoid(group_logits)
    threshold_tensor = torch.tensor(
        thresholds, dtype=probs.dtype, device=probs.device
    ).view(1, -1)
    above_threshold = probs > threshold_tensor
    any_above = above_threshold.any(dim=-1)

    masked_logits = group_logits.masked_fill(~above_threshold, float("-inf"))
    winners = masked_logits.argmax(dim=-1) + 1

    preds = torch.zeros(group_logits.shape[0], dtype=torch.long, device=group_logits.device)
    preds[any_above] = winners[any_above]
    return preds


def predict_grouped_actions(
    logits: torch.Tensor,
    move_threshold: float = 0.5,
    scroll_threshold: float = 0.5,
    click_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict movement/click group classes from action logits."""
    movement_preds = predict_group_action(
        logits[:, :2], (move_threshold, scroll_threshold)
    )
    click_preds = predict_group_action(logits[:, 2:], (click_threshold, click_threshold))
    return movement_preds, click_preds


def compute_pos_weights(dataloader: DataLoader, num_actions: int = 4) -> torch.Tensor:
    """Compute positive class weights for BCEWithLogitsLoss.

    Returns:
        Tensor of pos_weight for each action (num_negatives / num_positives).
    """
    all_actions = torch.cat([batch["actions"] for batch in dataloader])
    pos_counts = all_actions.sum(dim=0)
    neg_counts = len(all_actions) - pos_counts

    return neg_counts / (pos_counts + 1e-8)


def group_targets(
    action_targets: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """Convert two binary targets to grouped class labels.

    Output labels are:
        0 = nothing, 1 = first action, 2 = second action
    """
    group = action_targets[:, start:end]
    return torch.where(
        group.any(dim=-1),
        group.argmax(dim=-1) + 1,
        torch.zeros(group.shape[0], dtype=torch.long, device=group.device),
    )


def compute_group_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
) -> dict[str, float | torch.Tensor | list[str]]:
    """Compute accuracy/F1/confusion for one grouped classifier."""
    num_classes = len(class_names)
    metrics: dict[str, float | torch.Tensor | list[str]] = {
        "accuracy": (preds == targets).float().mean().item(),
    }

    preds_onehot = torch.nn.functional.one_hot(preds, num_classes).bool()
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes).bool()

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

    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.long, device=preds.device
    )
    indices = targets * num_classes + preds
    confusion.view(-1).scatter_add_(
        0, indices, torch.ones_like(indices, dtype=torch.long)
    )
    metrics["confusion_matrix"] = confusion
    metrics["class_names"] = class_names
    return metrics


def save_grouped_confusion_matrices(
    movement_cm: torch.Tensor,
    movement_class_names: list[str],
    click_cm: torch.Tensor,
    click_class_names: list[str],
    save_path: Path,
) -> None:
    """Save movement/click confusion matrices in a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrices = [
        (movement_cm, movement_class_names, "Movement Group Confusion Matrix"),
        (click_cm, click_class_names, "Click Group Confusion Matrix"),
    ]
    for ax, (cm, class_names, title) in zip(axes, matrices, strict=True):
        cm_np = cm.cpu().numpy()
        row_sums = cm_np.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm_np, row_sums, where=row_sums != 0) * 100

        im = ax.imshow(cm_pct, cmap="Blues")
        ax.set_title(title)
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


def plot_density(
    ax,
    values: np.ndarray,
    label: str,
    color: str,
    x_range: tuple[float, float] | None = None,
) -> bool:
    """Plot KDE if stable, otherwise fallback to a density histogram."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return False

    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min = float(clean.min())
        x_max = float(clean.max())
        if np.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0
        else:
            pad = 0.05 * (x_max - x_min)
            x_min -= pad
            x_max += pad

    if clean.size >= 2 and np.std(clean) > 1e-8:
        try:
            kde = gaussian_kde(clean)
            x = np.linspace(x_min, x_max, 200)
            y = kde(x)
            ax.plot(x, y, color=color, linewidth=2, label=label)
            return True
        except (np.linalg.LinAlgError, ValueError):
            pass

    bins = np.linspace(x_min, x_max, 30) if x_range is not None else 30
    ax.hist(clean, bins=bins, density=True, alpha=0.35, color=color, label=label)
    return True


def save_action_probability_density_plot(
    action_probs: np.ndarray,
    action_targets: np.ndarray,
    action_names: list[str],
    threshold: float,
    save_path: Path,
) -> None:
    """Save per-action probability density for negative vs positive labels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = np.asarray(axes).flatten()

    for i, action_name in enumerate(action_names):
        ax = axes[i]
        probs = action_probs[:, i]
        targets = action_targets[:, i] > 0.5

        has_data = False
        has_data |= plot_density(
            ax,
            probs[~targets],
            label="negative label",
            color="#1f77b4",
            x_range=(0.0, 1.0),
        )
        has_data |= plot_density(
            ax,
            probs[targets],
            label="positive label",
            color="#ff7f0e",
            x_range=(0.0, 1.0),
        )

        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"threshold={threshold:.1f}",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_title(action_name)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        if not has_data:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    for i in range(len(action_names), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Action probability density by label", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_zero_nonzero_dxdy_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    group_name: str,
    component_names: tuple[str, str],
    save_path: Path,
) -> None:
    """Save two-subplot dxdy distributions split by zero vs non-zero actual movement."""
    zero_mask = np.isclose(actual[:, 0], 0.0, atol=1e-6) & np.isclose(
        actual[:, 1], 0.0, atol=1e-6
    )
    split_defs = [
        ("Not moving (dx=0 and dy=0)", zero_mask),
        ("Moving (dx!=0 or dy!=0)", ~zero_mask),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (title, mask) in zip(axes, split_defs, strict=False):
        has_data = False
        has_data |= plot_density(
            ax,
            actual[mask].reshape(-1),
            label="actual",
            color="#1f77b4",
        )
        has_data |= plot_density(
            ax,
            predicted[mask].reshape(-1),
            label="predicted",
            color="#ff7f0e",
        )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"{title} (n={int(mask.sum())})")
        ax.set_xlabel(f"{component_names[0]}/{component_names[1]} value")
        ax.set_ylabel("Density")
        if not has_data:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()

    fig.suptitle(f"{group_name} dxdy distribution: actual vs predicted", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    action_loss_fn: nn.Module,
    dxdy_mean: torch.Tensor,
    dxdy_std: torch.Tensor,
) -> dict[str, object]:
    """Evaluate model on validation set."""
    model.eval()
    dxdy_loss_fn = nn.MSELoss()

    all_movement_preds = []
    all_movement_targets = []
    all_click_preds = []
    all_click_targets = []
    all_action_probs = []
    all_action_targets = []
    all_dxdy_pred_denorm = []
    all_dxdy_targets_denorm = []
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

        action_probs = torch.sigmoid(action_logits)
        dxdy_pred_denorm = dxdy_pred * dxdy_std + dxdy_mean
        dxdy_targets_denorm = dxdy_targets * dxdy_std + dxdy_mean

        movement_preds, click_preds = predict_grouped_actions(action_logits)
        movement_targets = group_targets(action_targets, start=0, end=2)
        click_targets = group_targets(action_targets, start=2, end=4)
        all_movement_preds.append(movement_preds)
        all_movement_targets.append(movement_targets)
        all_click_preds.append(click_preds)
        all_click_targets.append(click_targets)
        all_action_probs.append(action_probs.detach().cpu())
        all_action_targets.append(action_targets.detach().cpu())
        all_dxdy_pred_denorm.append(dxdy_pred_denorm.detach().cpu())
        all_dxdy_targets_denorm.append(dxdy_targets_denorm.detach().cpu())

    movement_preds = torch.cat(all_movement_preds)
    movement_targets = torch.cat(all_movement_targets)
    click_preds = torch.cat(all_click_preds)
    click_targets = torch.cat(all_click_targets)

    metrics: dict[str, object] = {
        "action_loss": total_action_loss / len(dataloader),
        "dxdy_loss": total_dxdy_loss / len(dataloader),
    }

    movement_metrics = compute_group_metrics(
        movement_preds, movement_targets, MOVEMENT_CLASS_NAMES
    )
    for key, value in movement_metrics.items():
        metrics[f"movement_{key}"] = value

    click_metrics = compute_group_metrics(click_preds, click_targets, CLICK_CLASS_NAMES)
    for key, value in click_metrics.items():
        metrics[f"click_{key}"] = value

    metrics["action_probs"] = torch.cat(all_action_probs, dim=0).numpy()
    metrics["action_targets"] = torch.cat(all_action_targets, dim=0).numpy()
    metrics["dxdy_pred_denorm"] = torch.cat(all_dxdy_pred_denorm, dim=0).numpy()
    metrics["dxdy_targets_denorm"] = torch.cat(all_dxdy_targets_denorm, dim=0).numpy()

    return metrics


def save_loss_plots(
    train_losses: dict[str, list[float]],
    val_losses: dict[str, list[float]],
    val_epochs: list[int],
    save_path: Path,
) -> None:
    """Save train/val loss curves as a figure with 3 subplots."""
    titles = [("action", "Action Loss"), ("dxdy", "dxdy Loss"), ("all", "Total Loss")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (key, title) in zip(axes, titles):
        train_epochs = range(1, len(train_losses[key]) + 1)
        ax.plot(train_epochs, train_losses[key], label="train")
        ax.plot(val_epochs, val_losses[key], label="val", marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    duration_s = total_data_duration_s(dataset)
    logger.info(
        f"Total data duration: {duration_s:.2f}s ({duration_s / 60:.2f} min, {duration_s / 3600:.2f} h)"
    )

    # Log normalization statistics
    logger.info(f"dxdy normalization - mean: {dataset.dxdy_mean.tolist()}")
    logger.info(f"dxdy normalization - std: {dataset.dxdy_std.tolist()}")

    # Split into train and eval per session (front=train, back=eval)
    train_idx, eval_idx, stats = split_tail(dataset, cfg.training.eval_split)
    for s in stats:
        logger.info(
            f"Session {s['i']:02d} split: total={s['n']} train={s['tr']} eval={s['ev']}"
        )
    train_dataset = Subset(dataset, train_idx)
    eval_dataset = Subset(dataset, eval_idx)
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
    scheduler = None
    if cfg.get("scheduler") is not None:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
        logger.info(f"Using LR scheduler: {scheduler.__class__.__name__}")
    else:
        logger.info("No LR scheduler configured")
    logger.info(
        f"Initial learning rate(s): {[group['lr'] for group in optimizer.param_groups]}"
    )

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
    dxdy_mean = torch.tensor(dataset.dxdy_mean, dtype=torch.float32, device=device)
    dxdy_std = torch.tensor(dataset.dxdy_std, dtype=torch.float32, device=device)

    # Loss histories for plotting
    train_losses = {"action": [], "dxdy": [], "all": []}
    val_losses = {"action": [], "dxdy": [], "all": []}
    val_epochs = []

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

        train_losses["action"].append(avg_action_loss)
        train_losses["dxdy"].append(avg_dxdy_loss)
        train_losses["all"].append(avg_loss)

        logger.info(
            f"Epoch {epoch + 1:3d}/{cfg.training.num_epochs} | "
            f"Loss: {avg_loss:.4f} (action: {avg_action_loss:.4f}, dxdy: {avg_dxdy_loss:.4f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Evaluation
        if (epoch + 1) % cfg.training.eval_interval == 0:
            metrics = evaluate(
                model, eval_loader, device, action_loss_fn, dxdy_mean, dxdy_std
            )
            logger.info(
                f"[Eval] Action Loss: {metrics['action_loss']:.4f} | "
                f"dxdy Loss: {metrics['dxdy_loss']:.4f}"
            )
            logger.info(
                f"       Movement Acc: {metrics['movement_accuracy']:.4f} | "
                f"F1 (macro): {metrics['movement_f1_macro']:.4f} | "
                f"F1 nothing: {metrics['movement_f1_nothing']:.4f} | "
                f"F1 move: {metrics['movement_f1_move']:.4f} | "
                f"F1 scroll: {metrics['movement_f1_scroll']:.4f}"
            )
            logger.info(
                f"       Click Acc: {metrics['click_accuracy']:.4f} | "
                f"F1 (macro): {metrics['click_f1_macro']:.4f} | "
                f"F1 nothing: {metrics['click_f1_nothing']:.4f} | "
                f"F1 left: {metrics['click_f1_left']:.4f} | "
                f"F1 right: {metrics['click_f1_right']:.4f}"
            )
            save_grouped_confusion_matrices(
                metrics["movement_confusion_matrix"],
                metrics["movement_class_names"],
                metrics["click_confusion_matrix"],
                metrics["click_class_names"],
                plots_dir / f"confusion_epoch_{epoch + 1:04d}.png",
            )

            val_epochs.append(epoch + 1)
            val_losses["action"].append(metrics["action_loss"])
            val_losses["dxdy"].append(metrics["dxdy_loss"])
            val_losses["all"].append(
                metrics["action_loss"] + metrics["dxdy_loss"]
            )
            save_loss_plots(
                train_losses, val_losses, val_epochs,
                plots_dir / "loss_curves.png",
            )
            save_action_probability_density_plot(
                metrics["action_probs"],
                metrics["action_targets"],
                ACTION_NAMES,
                threshold=0.5,
                save_path=plots_dir / f"action_density_epoch_{epoch + 1:04d}.png",
            )
            save_zero_nonzero_dxdy_plot(
                actual=metrics["dxdy_targets_denorm"][:, :2],
                predicted=metrics["dxdy_pred_denorm"][:, :2],
                group_name="Move",
                component_names=("dx", "dy"),
                save_path=plots_dir / f"move_dxdy_epoch_{epoch + 1:04d}.png",
            )
            save_zero_nonzero_dxdy_plot(
                actual=metrics["dxdy_targets_denorm"][:, 2:],
                predicted=metrics["dxdy_pred_denorm"][:, 2:],
                group_name="Scroll",
                component_names=("scroll_dx", "scroll_dy"),
                save_path=plots_dir / f"scroll_dxdy_epoch_{epoch + 1:04d}.png",
            )

            model.train()

        if scheduler is not None:
            scheduler.step()

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
