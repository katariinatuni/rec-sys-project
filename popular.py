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


import argparse
import numpy as np
from recsys.data import load_ratings
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items
from recsys.metrics import precision_at_k, recall_at_k, ndcg_at_k, catalogue_coverage

def evaluate_and_print(recs, target, items, variant_name):
    """This function calculates the metrics and prints them out."""
    # Converts items into a list for the entire catalog.
    full_catalogue = items.tolist()
    
    # Groups the predictions and the test set into dictionaries.
    all_predictions_dict = recs.groupby("user_id")["item_id"].apply(list).to_dict()
    test_ground_truth_dict = target.groupby("user_id")["item_id"].apply(list).to_dict()

    # Lists for the metric values ​​received by each individual user.
    all_precisions = []
    all_recalls = []
    all_ndcgs = []

    # Calculates metrics by user.
    for user_id, ground_truth in test_ground_truth_dict.items():
        if user_id in all_predictions_dict:
            predictions = all_predictions_dict[user_id]
            all_precisions.append(precision_at_k(predictions, ground_truth, k=10))
            all_recalls.append(recall_at_k(predictions, ground_truth, k=10))
            all_ndcgs.append(ndcg_at_k(predictions, ground_truth, k=10))

    # Calculates the final averages.
    mean_precision = np.mean(all_precisions) if all_precisions else 0.0
    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    mean_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0
    coverage = catalogue_coverage(list(all_predictions_dict.values()), full_catalogue, k=10)

    print(f"Variant / settings: {variant_name}")
    print(f"Precision@10:       {mean_precision:.4f}")
    print(f"Recall@10:          {mean_recall:.4f}")
    print(f"NDCG@10:            {mean_ndcg:.4f}")
    print(f"Catalogue coverage: {coverage:.4f}")

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

    
    # VARIANT 1: All-time popularity.
    recs_all_time = popular(fit, items, users, n=10, window_days=None)
    evaluate_and_print(recs_all_time, target, items, "All-time popularity")

    # VARIANT 2: Recent popularity.
    recs_recent = popular(fit, items, users, n=10, window_days=30)
    evaluate_and_print(recs_recent, target, items, "Recent window (30 days)")

if __name__ == "__main__":
    main()
