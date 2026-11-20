"""Markov-chain recommender (sequential paradigm, L4.1 slide 6).
Returns a DataFrame with columns user_id, item_id, rank (1 = best) and score.
Items the user has already rated are not recommended.

How it scores:
1. Each user's ratings are put in time order. Every rating on one submitted page shares a
   timestamp, so ties are broken by item_id to make the order deterministic.
2. Transitions a -> b are counted for b at most `window` positions after a in the same
   user's sequence (window = 1 is the classic first-order chain). With min_gap > 0, only
   pairs rated at least min_gap seconds apart count, which drops pairs from the same page.
3. Each row of counts is divided by its sum: P[a, b] = probability that b follows a.
4. The user's state is their last `history` items (None = all), the most recent weighted 1,
   the one before `decay`, then decay^2, ... The scores are the weighted average of those
   items' rows of P.
5. The scores are mixed with each item's share of all ratings:
   (1 - pop_weight) * scores + pop_weight * popularity. Both are probability
   distributions over items, so pop_weight is a plain mixing weight. This fills in items
   that never followed the user's items.

Defaults are the setting markov_tuning.py chose on validation (output/markov_chosen.json):
pairs at least one day apart, window 5,
the whole history without decay, 30 % popularity.

Run from the repository root (about 10 seconds):
    PYTHONPATH=src python -m recsys.markov --data-dir ../data/ml-1m
Options: --period val|test, the model settings below, and --symmetric (count b -> a too,
i.e. ignore the order: the control for "does direction matter?")."""

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from recsys.data import load_ratings
from recsys.metrics import evaluate, print_results
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items

SECONDS_PER_DAY = 24 * 60 * 60  # timestamps are unix seconds


def ordered_sequences(fit: pd.DataFrame, items: pd.Index) -> pd.DataFrame:
    """Ratings of candidate items in each user's time order.
    Adds col (the item's position in `items`, i.e. its matrix column) and
    pos (0 = the user's first rating, 1 = the next, ...)."""
    data = fit[fit["item_id"].isin(items)]  # only items from the items-list are selected
    data = data.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")  # same second: smaller item_id first
    data = data.assign(col=items.get_indexer(data["item_id"]),
                       pos=data.groupby("user_id").cumcount())
    return data.reset_index(drop=True)


def transition_matrix(seq: pd.DataFrame, n_items: int, window: int = 1,
                      symmetric: bool = False, min_gap: int = 0) -> csr_matrix:
    """(items x items) matrix: row a holds the probability of each item following a.
    Rows of items that nothing ever followed are all zero."""
    rows, cols = [], []
    by_user = seq.groupby("user_id")
    for d in range(1, window + 1):
        nxt = by_user["col"].shift(-d)              # the item d positions later, same user (NaN past the end)
        later = by_user["timestamp"].shift(-d)      # and when it was rated
        ok = nxt.notna() & (later - seq["timestamp"] >= min_gap)
        rows.append(seq.loc[ok, "col"].to_numpy())
        cols.append(nxt[ok].astype(int).to_numpy())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    if symmetric:  # count every pair both ways: the order is thrown away
        rows, cols = np.concatenate([rows, cols]), np.concatenate([cols, rows])

    counts = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_items, n_items))  # duplicate pairs are summed
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    inverse = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)  # 0 for empty rows, no division by zero
    return diags(inverse) @ counts


def recommend(fit: pd.DataFrame, items: pd.Index, users: pd.Index, n: int = 10,
              window: int = 5, history: int | None = None, decay: float = 1.0,
              pop_weight: float = 0.3, symmetric: bool = False,
              min_gap: int = SECONDS_PER_DAY) -> pd.DataFrame:

    seq = ordered_sequences(fit, items)
    P = transition_matrix(seq, len(items), window, symmetric, min_gap)

    # State matrix S (users x items): weight decay^j on the user's j-th most recent item.
    user_index = pd.Index(users)
    seq = seq[seq["user_id"].isin(users)]
    back = seq["pos"].groupby(seq["user_id"]).transform("max") - seq["pos"]  # 0 = most recent rating
    keep = back < history if history is not None else np.ones(len(seq), dtype=bool)
    S = csr_matrix((decay ** back[keep].to_numpy(),
                    (user_index.get_indexer(seq.loc[keep, "user_id"]), seq.loc[keep, "col"].to_numpy())),
                   shape=(len(users), len(items)))
    S = diags(1.0 / np.maximum(np.asarray(S.sum(axis=1)).ravel(), 1e-12)) @ S  # each user's weights sum to 1

    scores = np.asarray((S @ P).todense())  # (users x items) x (items x items)

    if pop_weight > 0:
        cols = items.get_indexer(fit["item_id"])
        popularity = np.bincount(cols[cols >= 0], minlength=len(items)).astype(float)  # all ratings, like popular()
        scores = (1 - pop_weight) * scores + pop_weight * popularity / popularity.sum()

    seen = np.zeros(scores.shape, dtype=bool)
    seen[user_index.get_indexer(seq["user_id"]), seq["col"].to_numpy()] = True
    scores[seen] = -np.inf  # setting the values of already seen movies to -inf

    top = np.argsort(-scores, axis=1, kind="stable")[:, :n]  # ties: smaller item_id first
    return pd.DataFrame({
        "user_id": np.repeat(user_index.to_numpy(), n),
        "item_id": items.to_numpy()[top].ravel(),
        "rank": np.tile(np.arange(1, n + 1), len(users)),
        "score": np.take_along_axis(scores, top, axis=1).ravel(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="val", choices=["val", "test"])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--history", type=int, default=None, help="last items used; default all")
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--pop-weight", type=float, default=0.3)
    parser.add_argument("--min-gap-days", type=float, default=1.0)
    parser.add_argument("--symmetric", action="store_true")
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)

    recs = recommend(fit, items, users, n=10, window=args.window, history=args.history,
                     decay=args.decay, pop_weight=args.pop_weight, symmetric=args.symmetric,
                     min_gap=int(args.min_gap_days * SECONDS_PER_DAY))
    print_results(f"Markov chain ({vars(args)})", evaluate(recs, target, users, items, k=10))


if __name__ == "__main__":
    main()
