"""Weighted hybrid recommender (Hybrid paradigm, L2.2 slide 10).
Combines User-based kNN and Markov chain scores linearly:
    score = alpha * user_knn_norm + (1 - alpha) * markov_norm
Scores of each model are min-max normalized per user over unseen candidate items."""

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from recsys.data import load_ratings
from recsys.knn_user import pearson_similarity
from recsys.markov import ordered_sequences, transition_matrix, SECONDS_PER_DAY
from recsys.metrics import evaluate, print_results
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Min-max normalize finite entries per row to, keeping -inf masked."""
    valid = np.isfinite(mat)
    masked = np.where(valid, mat, np.nan)
    row_min = np.nanmin(masked, axis=1, keepdims=True)
    row_max = np.nanmax(masked, axis=1, keepdims=True)
    denom = row_max - row_min
    norm = np.divide(masked - row_min, denom, out=np.zeros_like(masked), where=denom > 0)
    return np.where(valid, norm, -np.inf)


def _user_knn_scores(fit: pd.DataFrame, items: pd.Index, users: pd.Index,
                     k: int, gamma: int, pearson: np.ndarray | None) -> np.ndarray:
    """Raw (users x items) score matrix from User-based kNN, -inf for already rated."""
    data = fit[fit["item_id"].isin(items)].drop(["timestamp"], axis=1)
    pivot = data.pivot_table(index="item_id", columns="user_id", values="rating", aggfunc="mean")
    pivot = pivot.reindex(index=items)
    user_means = pivot.mean(axis=0)
    pivot_centered = pivot.subtract(user_means, axis=1).fillna(0.0)
    rated = pivot.notna().to_numpy().T

    training_matrix = csr_matrix(pivot_centered.values)
    if pearson is None:
        pearson = pearson_similarity(fit, items)

    n_corated = rated.astype(float) @ rated.T.astype(float)
    sim = pearson * (np.minimum(n_corated, gamma) / gamma)
    np.fill_diagonal(sim, -np.inf)
    indices = np.argsort(sim, axis=0, kind="stable")
    np.put_along_axis(sim, indices[:-k, :], 0, axis=0)

    scores = (training_matrix @ sim).T
    scores[rated] = -np.inf

    df = pd.DataFrame(scores, index=pivot_centered.columns, columns=items)
    return df.reindex(index=users, fill_value=-np.inf).to_numpy()


def _markov_scores(fit: pd.DataFrame, items: pd.Index, users: pd.Index,
                   window: int, pop_weight: float) -> np.ndarray:
    """Raw (users x items) score matrix from Markov chain, -inf for already rated."""
    seq = ordered_sequences(fit, items)
    P = transition_matrix(seq, len(items), window=window, min_gap=SECONDS_PER_DAY)

    user_index = pd.Index(users)
    seq_eval = seq[seq["user_id"].isin(users)]
    S = csr_matrix((np.ones(len(seq_eval)),
                    (user_index.get_indexer(seq_eval["user_id"]), seq_eval["col"].to_numpy())),
                   shape=(len(users), len(items)))
    S = diags(1.0 / np.maximum(np.asarray(S.sum(axis=1)).ravel(), 1e-12)) @ S

    scores = np.asarray((S @ P).todense())
    if pop_weight > 0:
        cols = items.get_indexer(fit["item_id"])
        popularity = np.bincount(cols[cols >= 0], minlength=len(items)).astype(float)
        scores = (1.0 - pop_weight) * scores + pop_weight * popularity / popularity.sum()

    seen = np.zeros(scores.shape, dtype=bool)
    seen[user_index.get_indexer(seq_eval["user_id"]), seq_eval["col"].to_numpy()] = True
    scores[seen] = -np.inf
    return scores


def recommend(fit: pd.DataFrame, items: pd.Index, users: pd.Index,
              alpha: float = 0.5, n: int = 10,
              user_k: int = 50, user_gamma: int = 30,
              markov_window: int = 5, markov_pop_weight: float = 0.3,
              pearson: np.ndarray | None = None) -> pd.DataFrame:
    """Weighted linear combination of normalized User kNN and Markov scores."""
    S_user = _normalize_rows(_user_knn_scores(fit, items, users, user_k, user_gamma, pearson))
    S_markov = _normalize_rows(_markov_scores(fit, items, users, markov_window, markov_pop_weight))

    both_unseen = np.isfinite(S_user) & np.isfinite(S_markov)
    scores = np.where(both_unseen, alpha * S_user + (1.0 - alpha) * S_markov, -np.inf)

    top = np.argsort(-scores, axis=1, kind="stable")[:, :n]
    return pd.DataFrame({
        "user_id": np.repeat(users.to_numpy(), n),
        "item_id": items.to_numpy()[top].ravel(),
        "rank": np.tile(np.arange(1, n + 1), len(users)),
        "score": np.take_along_axis(scores, top, axis=1).ravel(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="val", choices=["val", "test"])
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)

    recs = recommend(fit, items, users, alpha=args.alpha, n=10)
    print_results(f"Weighted Hybrid (alpha={args.alpha})", evaluate(recs, target, users, items, k=10))


if __name__ == "__main__":
    main()
