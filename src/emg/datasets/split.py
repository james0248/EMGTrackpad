from torch.utils.data import ConcatDataset


def split_tail(
    ds: ConcatDataset, ratio: float
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    """Split each session into front train / back eval and return global indices."""
    train_idx: list[int] = []
    eval_idx: list[int] = []
    stats: list[dict[str, int]] = []

    offset = 0
    for i, session_ds in enumerate(ds.datasets):
        n = len(session_ds)
        k = int(n * (1 - ratio))

        train_idx.extend(range(offset, offset + k))
        eval_idx.extend(range(offset + k, offset + n))
        stats.append({"i": i, "n": n, "tr": k, "ev": n - k})

        offset += n

    return train_idx, eval_idx, stats
