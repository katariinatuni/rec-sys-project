"""Compare the Part 1 baselines on one period.
Runs every most-popular variant and item kNN at several k, and prints one table with
the mean metrics and each variant's NDCG difference to the best variant, with a 95%
paired-bootstrap interval over users.

Run from the repository root (about 10 seconds):
    PYTHONPATH=src python -m recsys.compare --data-dir ../data/ml-1m
Options: --period val|test (default val; run test only once, after the choices are fixed)
and --resamples N (default 5000)."""

import argparse

import numpy as np
import pandas as pd

from recsys.data import load_ratings
from recsys.knn import recommend
from recsys.metrics import evaluate, per_user_metrics
from recsys.popular import popular, WINDOWS_DAYS
from recsys.split import (temporal_split, fit_and_target, evaluated_users, kept_items,
                          RELEVANT_MIN_RATING)

KNN_K = [20, 30, 50]  # neighbourhood sizes compared on validation

def bootstrap_diff(a: np.ndarray, b: np.ndarray, resamples: int = 5000,
                   seed: int = 0) -> tuple[float, float, float]:
    """
    Mean of a - b and its 95% interval, from resampling users with replacement.
    a and b hold one score per user, in the same user order, so each resample
    compares the two variants on exactly the same users (a paired bootstrap).
    """
    diff = a - b
    rng = np.random.default_rng(seed)  # same seed, same resamples for every comparison
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    low, high = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="val", choices=["val", "test"])
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)

    # Every variant: (model, setting, counting, recommendations).
    variants = [("Item kNN", f"k = {k}", "star ratings", recommend(fit, items, users, k=k))
                for k in KNN_K]
    for window_days in WINDOWS_DAYS:
        for min_rating in [None, RELEVANT_MIN_RATING]:
            variants.append(("Popular",
                             "all-time" if window_days is None else f"{window_days} days",
                             "all ratings" if min_rating is None else f"ratings >= {min_rating}",
                             popular(fit, items, users, window_days=window_days, min_rating=min_rating)))

    rows, ndcg_per_user = [], []
    for model, setting, counting, recs in variants:
        rows.append({"model": model, "setting": setting, "counting": counting,
                     **evaluate(recs, target, users, items, k=10)})
        ndcg_per_user.append(per_user_metrics(recs, target, users, k=10)["NDCG@10"].to_numpy())

    # Compare every variant with the one that has the highest mean NDCG.
    best = int(np.argmax([scores.mean() for scores in ndcg_per_user]))
    for row, scores in zip(rows, ndcg_per_user):
        diff, low, high = bootstrap_diff(scores, ndcg_per_user[best], args.resamples)
        row.update({"NDCG vs best": diff, "CI low": low, "CI high": high,
                    "clearly worse": high < 0})  # whole interval below zero

    print(f"{args.period}: {len(users)} users, best = "
          f"{rows[best]['model']} {rows[best]['setting']} {rows[best]['counting']}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
