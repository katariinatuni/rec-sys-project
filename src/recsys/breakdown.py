"""Disaggregated results: the chosen variant of each model, overall and broken down by group.

Two breakdowns:
1) User activity: users with at most / more than the median number of ratings in the fit data
2) Test time: held-out ratings before / from the median timestamp of the held-out ratings. 
    Each user is scored only on their relevant ratings in that half
The variants are the ones results.py chooses on validation. 
Each row's NDCG@10 gets a 95% bootstrap interval over the group's users (seed 0).

Run from the repository root:
    PYTHONPATH=src python -m recsys.breakdown --data-dir ../data/ml-1m
Options: --period val|test (default test), --resamples N (default 5000) and --out-dir
(default output; compare_val.csv is read from there and breakdown_<period>.csv written there)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_ratings
from recsys.knn import recommend
from recsys.knn_user import recommend as recommend_user
from recsys.metrics import per_user_metrics, catalogue_coverage, ranked_lists
from recsys.popular import popular
from recsys.results import choose, MODEL_NAMES
from recsys.split import (temporal_split, fit_and_target, evaluated_users, kept_items,
                          RELEVANT_MIN_RATING)


def run_variant(model, setting, counting, fit, items, users):
    """Recommendations of one variant, from the names compare.py gives it."""
    if model == "Item kNN":
        # setting is like "adjusted_cosine, k = 30"
        similarity, k = setting.split(", k = ")
        return recommend(fit, items, users, k=int(k), similarity=similarity)

    if model == "User kNN":
        # setting is like "pearson, gamma = 25, k = 50"
        rest, k = setting.split(", k = ")
        gamma = rest.split(", gamma = ")[1]
        return recommend_user(fit, items, users, k=int(k), gamma=int(gamma))

    # Popular: setting is like "14 days" or "all-time", counting like "ratings >= 4" or "all ratings"
    if setting == "all-time":
        window_days = None
    else:
        window_days = int(setting.split()[0])
    if counting == "all ratings":
        min_rating = None
    else:
        min_rating = RELEVANT_MIN_RATING
    return popular(fit, items, users, window_days=window_days, min_rating=min_rating)


def bootstrap_interval(scores, resamples, seed=0):
    """95% interval of the mean score, from resampling users with replacement."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        sample = rng.choice(scores, size=len(scores), replace=True)
        means.append(sample.mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return low, high


def make_row(model_name, dimension, group, definition, recs, target, group_users, items,
             resamples, note):
    """One row of the template: one model, one breakdown dimension, one group."""
    # Precision, recall and NDCG for every user in the group, one row per user.
    scores = per_user_metrics(recs, target, group_users, k=10)

    # Catalogue coverage of the lists recommended to this group's users only.
    group_recs = recs[recs["user_id"].isin(group_users)]
    coverage = catalogue_coverage(list(ranked_lists(group_recs).values()), items, k=10)

    low, high = bootstrap_interval(scores["NDCG@10"].to_numpy(), resamples)

    return {
        # The template's required columns in order.
        "Model": model_name,
        "Breakdown dimension": dimension,
        "Group": group,
        "Definition / threshold": definition,
        "Precision@10": scores["Precision@10"].mean(),
        "Recall@10": scores["Recall@10"].mean(),
        "NDCG@10": scores["NDCG@10"].mean(),
        "Catalogue coverage": coverage,
        "Seed(s)": "none (deterministic); bootstrap seed 0",
        "Notes": note,
        # Extra columns after the required ones:
        "Users": len(group_users),
        "NDCG@10 CI low": low,
        "NDCG@10 CI high": high,
    }


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

    # User activity: how many ratings each evaluated user has in the fit data.
    activity = fit.groupby("user_id").size()
    activity = activity.reindex(users, fill_value=0)  # same users, same order as `users`
    median_activity = activity.median()
    low_users = users[activity <= median_activity]
    high_users = users[activity > median_activity]

    # Test time: split the held-out ratings at their median timestamp.
    times = target["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, unit="s")  # MovieLens timestamps are seconds since 1970
    cut = times.median()
    early_target = target[times < cut]
    late_target = target[times >= cut]

    # In each half, keep only the users who have a relevant held-out rating in that half.
    early_relevant = early_target[early_target["rating"] >= RELEVANT_MIN_RATING]
    late_relevant = late_target[late_target["rating"] >= RELEVANT_MIN_RATING]
    early_users = users[users.isin(early_relevant["user_id"])]
    late_users = users[users.isin(late_relevant["user_id"])]

    activity_low = f"<= {median_activity:.0f} fit ratings (median of evaluated users)"
    activity_high = f"> {median_activity:.0f} fit ratings"
    time_early = f"held-out ratings before {cut:%Y-%m-%d %H:%M} (median timestamp)"
    time_late = f"held-out ratings from {cut:%Y-%m-%d %H:%M}"
    half_note = "scored on the relevant held-out ratings in this half only"

    # The variant of each model chosen on validation, by the same rule as results.py.
    chosen = choose(pd.read_csv(out / "compare_val.csv"))

    rows = []
    for model, setting, counting in chosen:
        recs = run_variant(model, setting, counting, fit, items, users)
        name = MODEL_NAMES[model]
        rows.append(make_row(name, "Overall", "All users", "every evaluated user",
                             recs, target, users, items, args.resamples, ""))
        rows.append(make_row(name, "User activity", "Low", activity_low,
                             recs, target, low_users, items, args.resamples, ""))
        rows.append(make_row(name, "User activity", "High", activity_high,
                             recs, target, high_users, items, args.resamples, ""))
        rows.append(make_row(name, "Test time", "Early", time_early,
                             recs, early_target, early_users, items, args.resamples, half_note))
        rows.append(make_row(name, "Test time", "Late", time_late,
                             recs, late_target, late_users, items, args.resamples, half_note))

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.4f"))
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / f"breakdown_{args.period}.csv", index=False)


if __name__ == "__main__":
    main()