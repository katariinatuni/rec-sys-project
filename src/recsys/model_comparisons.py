"""Paired comparisons of the chosen models: every pair, NDCG@10 difference with a 95% interval.

The variants are the ones results.py chooses on validation, so run compare.py on validation first.
Each pair is compared by a paired bootstrap over users (the same function as in compare.py): both
models are scored on the same users, so each user's difference counts.

Run from the repository root:
    PYTHONPATH=src python -m recsys.model_comparisons --data-dir ../data/ml-1m
Options: --period val|test (default test), --resamples N (default 5000) and --out-dir (default
output; compare_val.csv is read from there and comparisons_<period>.csv written there)."""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from recsys.breakdown import run_variant
from recsys.compare import bootstrap_diff
from recsys.data import load_ratings
from recsys.metrics import per_user_metrics
from recsys.results import choose, MODEL_NAMES
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="test", choices=["val", "test"])
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    out = Path(args.out_dir)

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)
    chosen = choose(pd.read_csv(out / "compare_val.csv"))  # variants chosen on validation

    # 1. Run every chosen variant once and keep the NDCG@10 of every user.
    ndcg = {}  # model name -> NDCG@10 of every user, in the order of `users`
    for model, setting, counting in chosen:
        recs = run_variant(model, setting, counting, fit, items, users)  # train and recommend
        ndcg[MODEL_NAMES[model]] = per_user_metrics(recs, target, users, k=10)["NDCG@10"].to_numpy()

    # 2. One row per pair of models, e.g. (Item-based kNN, User-based kNN).
    rows = []
    for model_a, model_b in combinations(ndcg, 2):  # every pair once
        diff, low, high = bootstrap_diff(ndcg[model_a], ndcg[model_b], args.resamples)
        rows.append({"Model A": model_a,
                     "Model B": model_b,
                     "NDCG@10 A": ndcg[model_a].mean(),
                     "NDCG@10 B": ndcg[model_b].mean(),
                     "Difference (A - B)": diff,
                     "95% CI low": low,
                     "95% CI high": high,
                     "Clearly different": low > 0 or high < 0,  # the interval does not include zero
                     "Users": len(users)})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))
    table.round(4).to_csv(out / f"comparisons_{args.period}.csv", index=False)


if __name__ == "__main__":
    main()