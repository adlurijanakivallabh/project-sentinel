import torch

try:
    from . import config
except ImportError:
    import config


def block_split(n_windows, block_size=None, val_size=None, test_size=None, seed=None):
    if block_size is None:
        block_size = config.WINDOW_BLOCK_SIZE
    if val_size is None:
        val_size = config.VAL_SIZE
    if test_size is None:
        test_size = config.TEST_SIZE
    if seed is None:
        seed = config.RANDOM_STATE

    n_blocks = n_windows // block_size
    remainder = n_windows % block_size

    torch.manual_seed(seed)
    block_perm = torch.randperm(n_blocks)

    train_end = int(n_blocks * (1 - val_size - test_size))
    val_end = int(n_blocks * (1 - test_size))

    train_blocks = block_perm[:train_end]
    val_blocks = block_perm[train_end:val_end]
    test_blocks = block_perm[val_end:]

    def blocks_to_indices(blocks):
        idx = []
        for b in blocks:
            start = b.item() * block_size
            end = min(start + block_size, n_windows)
            idx.extend(range(start, end))
        return torch.tensor(idx, dtype=torch.long)

    train_idx = blocks_to_indices(train_blocks)
    val_idx = blocks_to_indices(val_blocks)
    test_idx = blocks_to_indices(test_blocks)

    if remainder > 0:
        remainder_idx = torch.arange(n_blocks * block_size, n_windows)
        train_idx = torch.cat([train_idx, remainder_idx])

    all_idx = torch.cat([train_idx, val_idx, test_idx])
    assert len(all_idx) == n_windows, \
        f"Split lost windows: {len(all_idx)} vs {n_windows}"
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, \
        "Train/val overlap detected!"
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, \
        "Train/test overlap detected!"
    assert len(set(val_idx.tolist()) & set(test_idx.tolist())) == 0, \
        "Val/test overlap detected!"

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    n = 105863
    train_idx, val_idx, test_idx = block_split(n)
    print(f"Total windows:  {n}")
    print(f"Train:          {len(train_idx)} ({len(train_idx)/n*100:.1f}%)")
    print(f"Val:            {len(val_idx)} ({len(val_idx)/n*100:.1f}%)")
    print(f"Test:           {len(test_idx)} ({len(test_idx)/n*100:.1f}%)")
    print(f"Sum:            {len(train_idx) + len(val_idx) + len(test_idx)}")
    print(f"No overlap:     OK")
