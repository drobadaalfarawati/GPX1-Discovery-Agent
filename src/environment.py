import numpy as np

from .models import fit_surrogate
from .policies import binary_entropy


class GPX1DiscoveryEnv:
    """
    Closed-loop GPX1 discovery environment.

    Action 0: exploit highest predicted activity.
    Action 1: explore highest predictive entropy.
    """

    def __init__(
        self,
        X_campaign,
        y_campaign,
        campaign_scaffolds,
        budget=40,
        activity_reward=2.0,
        novelty_reward=1.0,
        information_reward=0.5,
        information_scale=100.0,
        recent_window=5,
        random_state=42,
    ):
        self.X = X_campaign
        self.y = y_campaign
        self.scaffolds = np.asarray(campaign_scaffolds)
        self.budget = budget
        self.activity_reward = activity_reward
        self.novelty_reward = novelty_reward
        self.information_reward = information_reward
        self.information_scale = information_scale
        self.recent_window = recent_window
        self.random_state = random_state

    def reset(self, seed_idx, pool_idx):
        self.seed_idx = np.asarray(seed_idx, dtype=int)
        self.pool_idx = np.asarray(pool_idx, dtype=int)
        self.labeled = list(self.seed_idx.copy())
        self.available = list(self.pool_idx.copy())
        self.history = []
        self.step_count = 0

        seed_active_idx = self.seed_idx[self.y[self.seed_idx] == 1]
        self.known_active_scaffolds = set(
            self.scaffolds[seed_active_idx]
        )

        self.model = fit_surrogate(
            self.X,
            self.y,
            self.labeled,
            self.random_state,
        )
        return self._observation()

    def _observation(self):
        progress = self.step_count / self.budget

        if self.history:
            labels = [h["observed_label"] for h in self.history]
            cumulative_hit_rate = float(np.mean(labels))
            recent_hit_rate = float(
                np.mean(labels[-self.recent_window:])
            )
        else:
            cumulative_hit_rate = 0.0
            recent_hit_rate = 0.0

        if self.available:
            p = self.model.predict_proba(
                self.X[self.available]
            )[:, 1]
            mean_pool_entropy = float(binary_entropy(p).mean())
        else:
            mean_pool_entropy = 0.0

        labeled = np.asarray(self.labeled, dtype=int)
        known_active_idx = labeled[self.y[labeled] == 1]

        if len(known_active_idx):
            scaffold_diversity = (
                len(set(self.scaffolds[known_active_idx]))
                / len(known_active_idx)
            )
        else:
            scaffold_diversity = 0.0

        return np.array(
            [
                progress,
                cumulative_hit_rate,
                recent_hit_rate,
                mean_pool_entropy,
                scaffold_diversity,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        if action not in (0, 1):
            raise ValueError(
                "action must be 0 (exploit) or 1 (explore)"
            )

        p = self.model.predict_proba(
            self.X[self.available]
        )[:, 1]
        entropy = binary_entropy(p)

        if action == 0:
            selected_position = int(np.argmax(p))
            decision_type = "Exploit"
        else:
            selected_position = int(np.argmax(entropy))
            decision_type = "Explore"

        selected_idx = self.available[selected_position]
        selected_scaffold = self.scaffolds[selected_idx]

        keep = np.ones(len(self.available), dtype=bool)
        keep[selected_position] = False
        entropy_before = (
            float(entropy[keep].mean())
            if keep.any()
            else 0.0
        )

        observed_label = int(self.y[selected_idx])

        novel_active_scaffold = int(
            observed_label == 1
            and selected_scaffold not in self.known_active_scaffolds
        )

        self.labeled.append(selected_idx)
        self.available.remove(selected_idx)

        if observed_label == 1:
            self.known_active_scaffolds.add(selected_scaffold)

        self.model = fit_surrogate(
            self.X,
            self.y,
            self.labeled,
            self.random_state,
        )

        if self.available:
            p_after = self.model.predict_proba(
                self.X[self.available]
            )[:, 1]
            entropy_after = float(binary_entropy(p_after).mean())
        else:
            entropy_after = 0.0

        information_gain = entropy_before - entropy_after
        scaled_information_gain = float(
            np.tanh(
                self.information_scale
                * information_gain
            )
        )

        reward = (
            self.activity_reward * observed_label
            + self.novelty_reward * novel_active_scaffold
            + self.information_reward * scaled_information_gain
        )

        self.step_count += 1

        info = {
            "step": self.step_count,
            "action": action,
            "decision_type": decision_type,
            "selected_index": selected_idx,
            "predicted_probability": float(
                p[selected_position]
            ),
            "selected_entropy": float(
                entropy[selected_position]
            ),
            "observed_label": observed_label,
            "novel_active_scaffold": novel_active_scaffold,
            "information_gain": float(information_gain),
            "scaled_information_gain": scaled_information_gain,
            "reward": float(reward),
        }

        self.history.append(info)

        terminated = self.step_count >= self.budget
        return (
            self._observation(),
            float(reward),
            terminated,
            info,
        )
