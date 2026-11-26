"""MMR re-ranking on top of the weighted hybrid.

1. The hybrid gives every user a candidate list of its 100 best movies, with scores that are
   min-max normalised per user (between 0 and 1).
2. MMR builds the final top-10 one movie at a time. At each step it picks the candidate that
   maximises
       lamba * relevance - (1 - lambda) * (highest similarity to the movies already picked),
   where relevance is the hybrid's score and similarity is the cosine of the genre vectors.
   lambda = 1 keeps the hybrid's own top-10, and smaller lambda values lead to more varied lists.

Run from the repository root:
    PYTHONPATH=src python -m recsys.rerank --data-dir ../data/ml-1m --lam 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recsys.data import load_ratings
from recsys.hybrid import recommend as recommend_hybrid
from recsys.metrics import evaluate, print_results
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items
from recsys.new_metrics import intra_list_diversity, gini, calibration_error

HYBRID_ALPHA = 0.4  # the hybrid's weight
N_CANDIDATES = 100  # how many of the hybrid's best movies MMR may choose from

 
def genre_vectors(data_dir) -> pd.DataFrame:
    """One row per movie (index item_id), one 0/1 column per genre, from movies.dat."""
    path = Path(data_dir) / "movies.dat"
    movies = pd.read_csv(path, sep="::", engine="python", encoding="latin-1", names=["item_id", "title", "genres"]) # Each line is id::title::genres, genres are separated by "|"
    movies = movies.set_index("item_id") 
    genres = movies["genres"].str.get_dummies(sep="|") # set 1 if the movie belongs to the genre, set 0 if not
    return genres
 
 
def candidates(fit, items, users, n=N_CANDIDATES) -> pd.DataFrame:
    """The hybrid's n best movies for every user, with their scores (user_id, item_id, rank, score)."""
    recs = recommend_hybrid(fit, items, users, alpha=HYBRID_ALPHA, n=n)
    has_score = np.isfinite(recs["score"])  # False for empty slots, when a user had too few candidates
    return recs[has_score]

def mmr_rerank(candidates, genres, lam, n=10) -> pd.DataFrame:
    """Each user's top-n chosen from their candidates by MMR (user_id, item_id, rank, score)."""
    rows = []
    for user, group in candidates.groupby("user_id", sort=False):
        items = group["item_id"].to_numpy()  # this user's candidates, best hybrid score first
        relevance = group["score"].to_numpy()  # the hybrid's scores, between 0 and 1
        similarity = cosine_similarity(genres.loc[items].to_numpy())  # candidates x candidates; the similarity is between 0-1, 1=exactly same genres, 0=no genres in common

        max_similarity = np.zeros(len(items))  # each candidate's highest similarity to the picked movies
        picked = np.zeros(len(items), dtype=bool)  # True for the candidates already in the list
        for rank in range(1, min(n, len(items)) + 1):
            mmr = lam * relevance - (1 - lam) * max_similarity
            mmr[picked] = -np.inf  # a movie can be picked only once
            best = int(np.argmax(mmr))  # ties: the one with the higher hybrid score
            rows.append({"user_id": user, "item_id": items[best], "rank": rank, "score": mmr[best]})
            picked[best] = True
            # The new movie may be more similar to some candidates than the earlier picks were.
            max_similarity = np.maximum(max_similarity, similarity[best])

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--period", default="val", choices=["val", "test"])
    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args()

    ratings = load_ratings(args.data_dir)
    split = temporal_split(ratings)
    fit, target = fit_and_target(split, args.period)
    users = evaluated_users(fit, target)
    items = kept_items(fit)
    genres = genre_vectors(args.data_dir)

    pool = candidates(fit, items, users)  # computed once; every lambda reuses it
    recs = mmr_rerank(pool, genres, lam=args.lam)
    print_results(f"Hybrid + MMR (lam = {args.lam})", evaluate(recs, target, users, items, k=10))

    print("ILD", intra_list_diversity(recs, genres))
    print("Gini", gini(recs, items))
    print("Calibration error", calibration_error(recs, fit, genres))


if __name__ == "__main__":
    main()