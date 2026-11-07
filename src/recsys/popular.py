"""Most popular baseline.
Returns a DataFrame with columns user_id, item_id, rank (1 = best) and score.
Items the user has already rated are not recommended.

Run from the repository root to compare all popularity variants (about 3 seconds):
    PYTHONPATH=src python -m recsys.popular --data-dir ../data/ml-1m"""

import argparse

import pandas as pd

from recsys.data import load_ratings
from recsys.metrics import evaluate
from recsys.split import (temporal_split, fit_and_target, evaluated_users, kept_items,
                          RELEVANT_MIN_RATING)

SECONDS_PER_DAY = 24 * 60 * 60  # timestamps are unix seconds
WINDOWS_DAYS = [None, 7, 14, 30, 60]  # None = all-time; the rest are compared on validation

def popular(fit: pd.DataFrame, items: pd.Index, users: pd.Index,
            n: int = 10, window_days: int | None = None,
            min_rating: int | None = None) -> pd.DataFrame:

    data = fit[fit["item_id"].isin(items)]  # only items from the items-list are selected

    if window_days is not None:
        start = fit["timestamp"].max() - window_days * SECONDS_PER_DAY  # window is counted from the last rating in fit
        data = data[data["timestamp"] > start]

    if min_rating is not None:
        data = data[data["rating"] >= min_rating]  # count only positive ratings; None counts every rating

    counts = data["item_id"].value_counts().reindex(items, fill_value=0)  # number of (positive) ratings per item
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")  # ties: smaller item_id first

    seen = fit.groupby("user_id")["item_id"].apply(set)  # all movies the user has rated in fit, at any rating

    rows = []

    for user in users:
        user_seen = seen.get(user, set())
        unseen = counts[~counts.index.isin(user_seen)]  # already seen movies are removed
        top = unseen.head(n)

        for rank, (item_id, score) in enumerate(top.items(), start=1):
            rows.append({"user_id": user, "item_id": item_id, "rank": rank, "score": score})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="val", choices=["val", "test"])
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)

    # Two choices from the project brief, every combination:
    # all-time or a recent window of several lengths, counting all ratings or only positive ones.
    rows = []
    for window_days in WINDOWS_DAYS:
        for min_rating in [None, RELEVANT_MIN_RATING]:
            recs = popular(fit, items, users, n=10, window_days=window_days, min_rating=min_rating)
            rows.append({"window": "all-time" if window_days is None else f"{window_days} days",
                         "counting": "all ratings" if min_rating is None else f"ratings >= {min_rating}",
                         **evaluate(recs, target, users, items, k=10)})
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()

