"""Compare the baselines, user kNN and the Markov chain on one period.
Runs every most-popular variant, item kNN (adjusted cosine and cosine) at several k,
user kNN (Pearson with significance weighting) at several k and gamma and the Markov
chain at several windows and popularity weights, and prints one table with the mean
metrics and each variant's NDCG difference to the best variant, with a 95%
paired-bootstrap interval over users.

Run from the repository root (about 1-2 min):
    PYTHONPATH=src python -m recsys.compare --data-dir ../data/ml-1m
Options: --period val|test (default val; run test only once, after the choices are fixed),
--resamples N (default 5000) and --out-dir (default output; the table is also saved there
as compare_<period>.csv)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_ratings
from recsys.knn import recommend
from recsys.metrics import evaluate, per_user_metrics
from recsys.popular import popular, WINDOWS_DAYS
from recsys.split import (temporal_split, fit_and_target, evaluated_users, kept_items,
                          RELEVANT_MIN_RATING)
from recsys.knn_user import recommend as recommend_user, pearson_similarity
from recsys.markov import recommend as recommend_markov

from recsys.hybrid import recommend as recommend_hybrid

KNN_K = [20, 30, 50]  # neighbourhood sizes compared on validation
USER_KNN_K = [20, 30, 50]  # neighbourhood sizes compared on validation
USER_KNN_GAMMA = [1, 30]  # significance weighting thresholds compared on validation (1 means no significance weighting)
# Markov chain: the two settings that still matter after the 84-setting search in markov_tuning.py.
# The others stay at recommend()'s defaults, which that search fixed (whole history, no decay, pairs >= 1 day apart).
MARKOV_WINDOW = [1, 5, 20]  # how many positions ahead a pair a -> b may be (1 = the classic first-order chain)
MARKOV_POP_WEIGHT = [0.0, 0.3]  # share of popularity mixed into the scores (0.0 = transitions only)
HYBRID_ALPHAS = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # weight of User kNN in alpha * User_kNN + (1 - alpha) * Markov; 6 values to match the validation search budget of the other models


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
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)

    # Every variant: (model, setting, counting, recommendations).
    variants = [("Item kNN", f"{sim}, k = {k}", "star ratings", recommend(fit, items, users, k=k, similarity=sim))
                for sim in ["adjusted_cosine", "cosine"] for k in KNN_K]
    pearson = pearson_similarity(fit, items)  # slow, so computed once and shared by all User kNN variants
    for gamma in USER_KNN_GAMMA:
        for k in USER_KNN_K:
            variants.append(("User kNN", f"pearson, gamma = {gamma}, k = {k}", "star ratings", recommend_user(fit, items, users, k=k, gamma=gamma, pearson=pearson)))
    for window in MARKOV_WINDOW:
        for pop_weight in MARKOV_POP_WEIGHT:
            variants.append(("Markov", f"window = {window}, pop_weight = {pop_weight}", "rating order",
                             recommend_markov(fit, items, users, window=window, pop_weight=pop_weight)))
    for alpha in HYBRID_ALPHAS:
        variants.append(("Hybrid", f"alpha = {alpha}", "normalized scores",
                         recommend_hybrid(fit, items, users, alpha=alpha, pearson=pearson)))
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
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(args.out_dir) / f"compare_{args.period}.csv", index=False)  # read by results.py


if __name__ == "__main__":
    main()