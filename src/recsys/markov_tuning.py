"""Tune the Markov chain on validation: a fixed grid, the same selection rule as Part 1.

Search space (84 settings, every one is run; about the size of the Part 1 item-kNN sweep):
    window      1, 5, 20          how many positions ahead a pair a -> b may be
    history     1, 5, 20, all     how many of the user's most recent items form the state
    decay       1.0, 0.8          weight of older items in the state (only when history > 1)
    pop_weight  0.0, 0.3          share of popularity mixed into the scores
    min_gap     0, 1 day          shortest time between a and b for the pair to count
history = 1 with decay 0.8 is left out: one item gets weight 1 whatever the decay.

Selection rule (results.py, Part 1): among the settings that are not clearly worse than the
best one (the 95 % paired-bootstrap interval of the NDCG@10 difference includes zero), take
the one with the highest catalogue coverage; NDCG@10 breaks ties.

Validation only. Writes output/markov_tuning_val.csv (every setting) and
output/markov_chosen.json (the chosen one: markov.py's defaults and compare.py's Markov grid
are built around it).

Run from the repository root (about two minutes):
    PYTHONPATH=src python -m recsys.markov_tuning --data-dir ../data/ml-1m
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.compare import bootstrap_diff
from recsys.data import load_ratings
from recsys.markov import recommend, SECONDS_PER_DAY
from recsys.metrics import evaluate, per_user_metrics
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items

WINDOWS = [1, 5, 20]
HISTORIES = [1, 5, 20, None]  # None = the whole history
DECAYS = [1.0, 0.8]
POP_WEIGHTS = [0.0, 0.3]
MIN_GAPS = [0, SECONDS_PER_DAY]


def grid() -> list[dict]:
    """Every setting in the search space, as keyword arguments for recommend()."""
    settings = []
    for window, history, decay, pop_weight, min_gap in itertools.product(
            WINDOWS, HISTORIES, DECAYS, POP_WEIGHTS, MIN_GAPS):
        if history == 1 and decay != 1.0:
            continue  # same model as decay 1.0
        settings.append(dict(window=window, history=history, decay=decay,
                             pop_weight=pop_weight, min_gap=min_gap))
    return settings


def choose(table: pd.DataFrame) -> int:
    """Row index of the chosen setting: not clearly worse, then most coverage, then NDCG."""
    qualifying = table[~table["clearly worse"]]
    return qualifying.sort_values(["Coverage", "NDCG@10"], ascending=False, kind="stable").index[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()

    fit, target = fit_and_target(temporal_split(load_ratings(args.data_dir)), "val")
    users, items = evaluated_users(fit, target), kept_items(fit)

    rows, ndcg_per_user = [], []
    for setting in grid():
        recs = recommend(fit, items, users, n=10, **setting)
        rows.append({**setting, **evaluate(recs, target, users, items, k=10)})
        ndcg_per_user.append(per_user_metrics(recs, target, users, k=10)["NDCG@10"].to_numpy())

    best = int(np.argmax([scores.mean() for scores in ndcg_per_user]))
    for row, scores in zip(rows, ndcg_per_user):
        diff, low, high = bootstrap_diff(scores, ndcg_per_user[best], args.resamples)
        row.update({"NDCG vs best": diff, "CI low": low, "CI high": high, "clearly worse": high < 0})

    table = pd.DataFrame(rows)
    chosen = choose(table)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "markov_tuning_val.csv", index=False)
    setting = {key: table.loc[chosen, key] for key in ["window", "history", "decay", "pop_weight", "min_gap"]}
    setting = {key: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for key, v in setting.items()}
    (out / "markov_chosen.json").write_text(json.dumps(setting, indent=2) + "\n")

    print(f"validation: {len(users)} users, {len(table)} settings, "
          f"{(~table['clearly worse']).sum()} not clearly worse than the best")
    print(table.sort_values("NDCG@10", ascending=False).head(15).to_string(index=False, float_format="%.4f"))
    print(f"\nbest NDCG@10: {table.loc[best, 'NDCG@10']:.4f}  chosen: {setting} "
          f"(NDCG@10 {table.loc[chosen, 'NDCG@10']:.4f}, coverage {table.loc[chosen, 'Coverage']:.1%})")


if __name__ == "__main__":
    main()
