"""Item-based kNN baseline.
Returns a dataFrame with columns user_id, item_id, rank (1 = best) and score.
Items the user has already rated are not recommended."""

import argparse

from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recsys.data import load_ratings
from recsys.metrics import evaluate, print_results
from recsys.split import (temporal_split, fit_and_target,
                          evaluated_users, kept_items)

def recommend(fit: pd.DataFrame, items: pd.Index, users: pd.Index,
              k: int = 50, n: int = 10) -> pd.DataFrame: 

    data = fit[fit["item_id"].isin(items)].drop(["timestamp"], axis=1) # only items from the items-list are selected

    pivot = data.pivot_table(index="item_id", columns="user_id", values="rating", aggfunc="mean") # converting matrix into a pivot table

    user_means = pivot.mean(axis=0) # rating means by user

    pivot_centered = pivot.subtract(user_means, axis=1).fillna(0) # mean centering by user and setting NaN to 0

    rated = pivot.notna().to_numpy().T   # (users x movies), True = user has already rated  

    user_ids = pivot_centered.columns
    item_ids = pivot_centered.index
    
    training_matrix = csr_matrix(pivot_centered.values) # convert dataframe to a scipy sparse matrix

    similarity_matrix = cosine_similarity(training_matrix)
    np.fill_diagonal(similarity_matrix, -np.inf) # sets to -inf the similarity of the movie with itself

    indices = np.argsort(similarity_matrix, axis=0, kind="stable") # sorts indices by values
    np.put_along_axis(similarity_matrix, indices[:-k, :], 0, axis=0) # leaves only the k nearest neighbors, and sets other values to 0

    recommendation_matrix = training_matrix.T @ similarity_matrix # (users x movies) x (movies x movies)
    
    recommendation_matrix[rated] = -np.inf # setting the values of already seen movies to -inf
    
    recommendation_df = pd.DataFrame(data=recommendation_matrix, index=user_ids, columns=item_ids)
    
    rows = []

    for user in users:
        if user not in recommendation_df.index:    
            continue
        user_series = recommendation_df.loc[user].sort_values(ascending=False, kind="stable")
        top_n = user_series.head(n)

        for rank, (item_id, score) in enumerate(top_n.items(), start=1):
            rows.append({"user_id": user, "item_id": item_id, "rank": rank, "score": score})

    results_df = pd.DataFrame(rows)

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="val", choices=["val", "test"])
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)

    recs = recommend(fit, items, users, k=args.k, n=10)
    print_results(f"Item kNN (k={args.k})", evaluate(recs, target, users, items, k=10))


if __name__ == "__main__":
    main()