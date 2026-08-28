import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def scaffold_split(X, y, scaffolds, test_size=0.20, random_state=42):
    """Create a scaffold-disjoint development/validation split."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, val_idx = next(
        splitter.split(X, y, groups=np.asarray(scaffolds))
    )
    return train_idx, val_idx


def create_seed_set(
    y_campaign,
    campaign_seed,
    n_actives=5,
    n_inactives=45,
):
    """Simulate a small historical labeled set."""
    rng = np.random.default_rng(campaign_seed)

    active_candidates = np.where(y_campaign == 1)[0]
    inactive_candidates = np.where(y_campaign == 0)[0]

    seed_active_idx = rng.choice(
        active_candidates, size=n_actives, replace=False
    )
    seed_inactive_idx = rng.choice(
        inactive_candidates, size=n_inactives, replace=False
    )

    seed_idx = np.concatenate([seed_active_idx, seed_inactive_idx])
    rng.shuffle(seed_idx)

    all_idx = np.arange(len(y_campaign))
    pool_idx = np.setdiff1d(all_idx, seed_idx)
    return seed_idx, pool_idx
