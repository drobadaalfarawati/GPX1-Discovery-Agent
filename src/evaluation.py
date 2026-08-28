import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def top_k_screening_table(ranked, budgets, active_rate):
    rows = []
    for k in budgets:
        top = ranked.head(k)
        hits = int(top["true_label"].sum())
        random_expected = k * active_rate
        rows.append({
            "budget": k,
            "actives_found": hits,
            "hit_rate": hits / k,
            "random_expected": random_expected,
            "enrichment": hits / random_expected if random_expected else np.nan,
        })
    return pd.DataFrame(rows)


def completed_campaign_metrics(
    history,
    seed_idx,
    X_campaign,
    y_campaign,
    campaign_scaffolds,
    X_validation,
    y_validation,
    fit_surrogate_fn,
):
    hits = int(history["observed_label"].sum())

    active_selected_idx = (
        history.loc[
            history["observed_label"] == 1,
            "selected_index",
        ]
        .astype(int)
        .to_numpy()
    )
    active_scaffolds = len(
        set(campaign_scaffolds[active_selected_idx])
    )

    selected_idx = history["selected_index"].astype(int).to_numpy()
    final_labeled_idx = np.concatenate([seed_idx, selected_idx])

    final_model = fit_surrogate_fn(
        X_campaign,
        y_campaign,
        final_labeled_idx,
    )
    p = final_model.predict_proba(X_validation)[:, 1]

    return {
        "hits": hits,
        "hit_rate": hits / len(history),
        "active_scaffolds": active_scaffolds,
        "validation_pr_auc": average_precision_score(
            y_validation,
            p,
        ),
    }
