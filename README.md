# GPX1 Discovery Agent

A compact demonstration of a **scientific learning layer** for small-molecule discovery using RDKit, active learning, scientific reward engineering, and reinforcement learning.

## Start here

For the complete end-to-end walkthrough, open:

`00_GPX1_interview_demo.ipynb.ipynb`

This notebook presents the full project story in one place: molecular modeling, scaffold-aware evaluation, closed-loop experiment selection, reward engineering, Q-learning, and reward ablation.

## Core question

> Can a scientific agent learn when to exploit predicted GPX1 activity and when to explore uncertain chemistry under a fixed experimental budget?

## Workflow

```text
GPX1 activity data
        ↓
RDKit + Morgan fingerprints
        ↓
Scaffold-aware surrogate model
        ↓
Closed-loop screening simulation
        ↓
Exploit / explore acquisition policies
        ↓
Scientific reward
        ↓
Q-learning meta-policy
        ↓
Repeated campaign evaluation
```

## Dataset

The curated development dataset contains:

* 9,304 compounds
* 155 actives
* 9,149 inactives
* 1.67% active rate
* 5,825 unique scaffolds

The raw CSV is intentionally not committed until the exact PubChem assay provenance and curation procedure are fully documented.

## Key results

### Molecular baseline

A class-balanced logistic regression model trained on 2,048-bit Morgan fingerprints produced:

| Metric                   | Result |
| ------------------------ | -----: |
| ROC-AUC                  |  0.813 |
| PR-AUC                   |  0.309 |
| Random PR baseline       |  0.019 |
| PR-AUC / random baseline |  16.2× |

At a screening budget of 40 compounds, the ranking recovered 13 actives versus approximately 0.76 expected from random screening.

### RL versus strong baseline

| Strategy          | Mean reward | Mean hits | Active scaffolds | Validation PR-AUC |
| ----------------- | ----------: | --------: | ---------------: | ----------------: |
| Pure exploitation |   **94.58** |  **33.9** |         **23.6** |             0.221 |
| Q-learning        |       90.23 |      31.9 |             22.5 |         **0.224** |
| Pure exploration  |       27.37 |      6.75 |              6.7 |             0.240 |

The learned Q-policy did **not** beat pure exploitation under the initial hit-heavy reward. Rather than retuning the reward until RL appeared superior, that result was treated as evidence about the objective itself.

### Reward ablation

Changing the reward changed the learned behavior. A learning-balanced objective caused substantially more exploration, but rewarding scaffold novelty did not increase scaffold discovery.

The key design insight was:

> **Uncertainty exploration is not the same as chemical-novelty exploration.**

Reward design and action-space design therefore need to be co-designed.

## Project components

* `00_GPX1_interview_demo.ipynb.ipynb` — complete end-to-end interview demo
* `01_model_development.ipynb` — molecular baseline and scaffold-aware evaluation
* `02_active_learning.ipynb` — closed-loop active-learning campaigns
* `03_rl_reward_engineering.ipynb` — reward design and reinforcement learning
* `featurization.py` — molecular representation
* `models.py` — surrogate models
* `campaign.py` — campaign utilities
* `environment.py` — discovery environment
* `evaluation.py` — evaluation utilities
* `policies.py` — acquisition policies

## Design lessons

* Use PR-AUC and screening enrichment instead of accuracy for severe class imbalance.
* Hold out molecular scaffolds to reduce analogue leakage.
* Validate acquisition functions analytically, not only empirically.
* Audit reward-component scale before RL training.
* Separate the scientific rubric from the numerical reward.
* Separate reward design from action-space design.
* Compare RL against strong simple baselines.
* Do not tune the reward merely to make RL win.

## Prototype limitations

* Binary activity labels rather than quantitative potency.
* Historical campaign seeds are intentionally stratified.
* Logistic-regression scores are not guaranteed to be calibrated probabilities.
* The current uncertainty signal is not rigorous epistemic uncertainty.
* The exploration action optimizes predictive uncertainty rather than explicit chemical novelty.
* The scaffold-held-out set was repeatedly inspected during development and is therefore a validation set, not a pristine final test set.
* Exact PubChem assay provenance should be documented before distributing the raw dataset.

## Future work

* Explicit scaffold-diversity acquisition actions
* Ensemble-based epistemic uncertainty
* Quantitative potency and multi-endpoint rewards
* Experimental cost and assay noise
* Batch experiment selection
* Richer state and action spaces
* PPO or neural value functions only when the environment complexity justifies them

## Central takeaway

The behavior of an autonomous scientific agent depends on the interaction between the **model, environment, actions, evaluation rubric, and reward function**. Designing that learning layer is itself a scientific problem.
