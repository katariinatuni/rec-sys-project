"""Most popular baseline.
Returns a DataFrame with columns user_id, item_id, rank (1 = best) and score.
Items the user has already rated are not recommended."""

import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60  # timestamps are unix seconds

def popular(fit: pd.DataFrame, items: pd.Index, users: pd.Index,
            n: int = 10, window_days: int = None) -> pd.DataFrame:

    data = fit[fit["item_id"].isin(items)]  # only items from the items-list are selected

    if window_days is not None:
        start = fit["timestamp"].max() - window_days * SECONDS_PER_DAY  # window is counted from the last rating in fit
        data = data[data["timestamp"] > start]

    counts = data["item_id"].value_counts().reindex(items, fill_value=0)  # number of ratings per item
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")  # ties: smaller item_id first

    seen = fit.groupby("user_id")["item_id"].apply(set)  # all movies the user has rated in fit

    rows = []

    for user in users:
        user_seen = seen.get(user, set())
        unseen = counts[~counts.index.isin(user_seen)]  # already seen movies are removed
        top = unseen.head(n)

        for rank, (item_id, score) in enumerate(top.items(), start=1):
            rows.append({"user_id": user, "item_id": item_id, "rank": rank, "score": score})

    return pd.DataFrame(rows)

