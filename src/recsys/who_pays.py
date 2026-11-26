"""Part 3 (Track A): who pays for diversity? The chosen lambda against lambda = 1, on test.

The lambda chosen on validation is read from mmr_chosen.json (written by mmr_sweep.py). Lambda = 0.5
is shown too, to see how the cost grows with a stronger setting. Both are compared with lambda = 1,
the hybrid's own top-10, on the same users and the same candidate pool.

1) Users: all users, then split by activity (at most / more than the median number of fit
   ratings, as in breakdown.py) and by taste breadth (at most / more than the median genre entropy
   of the user's fit ratings; low entropy = a history concentrated in few genres). Per group:
   NDCG@10, ILD and calibration error at both lambdas, with paired 95% bootstrap intervals for the
   NDCG@10 and calibration differences (seed 0).
2) Items: the share of recommendation slots that go to head items (the 10 % most-rated kept
   items), each genre's share of the slots, and the movies that lose and gain the most slots.

Run from the repository root, after mmr_sweep.py:
    PYTHONPATH=src python -m recsys.who_pays --data-dir ../data/ml-1m --out-dir output_part3
Options: --resamples N (default 5000) and --out-dir (default output; mmr_chosen.json is read from
there). Writes who_pays_users_test.csv, who_pays_items_test.csv, who_pays_genres_test_lam<lambda>.csv
and who_pays_movers_test_lam<lambda>.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recsys.compare import bootstrap_diff
from recsys.data import load_movies, load_ratings
from recsys.metrics import per_user_metrics
from recsys.mmr import candidates, genre_vectors, mmr_rerank
from recsys.new_metrics import alfa, calibration_error, genre_distribution, intra_list_diversity
from recsys.split import evaluated_users, fit_and_target, kept_items, temporal_split

REFERENCE = 1.0          # the unmodified hybrid
CONTRAST_LAMBDA = 0.5    # a stronger setting, to see how the cost grows
HEAD_SHARE = 0.10        # head items = the 10 % most-rated kept items


# Per-user versions of the metrics in new_metrics.py, which only return the mean over users.
# The bootstrap and the group means need one value per user; main() checks that the means agree.

def per_user_ild(recs, genres) -> pd.Series:
    """Mean pairwise genre distance (1 - cosine) inside each user's list."""
    values = {}
    for user, group in recs.groupby("user_id", sort=False):
        vectors = genres.loc[group["item_id"]].to_numpy()
        if len(vectors) < 2:
            continue  # a list with one movie has no pairs
        distance = 1 - cosine_similarity(vectors)
        values[user] = distance[np.triu_indices(len(vectors), k=1)].mean()
    return pd.Series(values)


def per_user_calibration(recs, p, genres) -> pd.Series:
    """Each user's KL(p || q~), with q~ = (1 - alfa) q + alfa p, as in calibration_error."""
    q = genre_distribution(recs, genres)
    p = p.loc[q.index]
    q_smooth = (1 - alfa) * q + alfa * p
    ratio = (p / q_smooth).where(p > 0, 1)  # genres with p = 0 add log(1) = 0
    return (p * np.log(ratio)).sum(axis=1)


def genre_entropy(p) -> pd.Series:
    """Shannon entropy (in nats) of each user's genre distribution: low = narrow taste."""
    return -(p * np.log(p.where(p > 0, 1))).sum(axis=1)


def per_user_scores(recs, target, users, p, genres) -> pd.DataFrame:
    """One row per evaluated user: NDCG@10, number of hits, ILD and calibration error."""
    accuracy = per_user_metrics(recs, target, users, k=10)
    return pd.DataFrame({
        "NDCG": accuracy["NDCG@10"],
        "hits": (accuracy["Precision@10"] * 10).round(),  # relevant movies in the top-10
        "ILD": per_user_ild(recs, genres).reindex(users),
        "KL": per_user_calibration(recs, p, genres).reindex(users),
    })


def group_rows(scores, groups, lam, resamples) -> list[dict]:
    """For each user group: the metrics at lambda = 1 and at lam, and the paired differences."""
    rows = []
    for dimension, labels in groups.items():
        for name in labels.unique():
            members = labels.index[labels == name]
            a, b = scores[lam].loc[members], scores[REFERENCE].loc[members]
            d_ndcg = bootstrap_diff(a["NDCG"].to_numpy(), b["NDCG"].to_numpy(), resamples)
            d_kl = bootstrap_diff(a["KL"].to_numpy(), b["KL"].to_numpy(), resamples)
            d_hits = a["hits"] - b["hits"]
            rows.append({
                "lambda": lam, "dimension": dimension, "group": name, "users": len(members),
                "NDCG ref": b["NDCG"].mean(), "NDCG": a["NDCG"].mean(),
                "dNDCG": d_ndcg[0], "dNDCG low": d_ndcg[1], "dNDCG high": d_ndcg[2],
                "rel dNDCG %": 100 * d_ndcg[0] / b["NDCG"].mean(),
                "ILD ref": b["ILD"].mean(), "ILD": a["ILD"].mean(),
                "KL ref": b["KL"].mean(), "KL": a["KL"].mean(),
                "dKL": d_kl[0], "dKL low": d_kl[1], "dKL high": d_kl[2],
                "users losing hits %": 100 * (d_hits < 0).mean(),
                "users gaining hits %": 100 * (d_hits > 0).mean(),
                "users KL worse %": 100 * (a["KL"] > b["KL"] + 1e-12).mean(),
            })
    return rows


def item_summary(recs, counts, head) -> dict:
    """How popular the recommended movies are: head share, mean popularity percentile, distinct."""
    slots = recs["item_id"]
    percentile = counts.rank(pct=True)  # popularity percentile within the kept items
    return {"head slot share": slots.isin(head).mean(),
            "mean popularity percentile": percentile.loc[slots].mean(),
            "distinct items": slots.nunique()}


def genre_slots(recs_ref, recs_lam, lam, p, genres) -> pd.DataFrame:
    """Each genre's share of the recommendation slots (a movie with n genres gives 1/n to each)."""
    shares = genres.div(genres.sum(axis=1), axis=0)
    slots_ref = shares.loc[recs_ref["item_id"]].sum()
    slots_lam = shares.loc[recs_lam["item_id"]].sum()
    table = pd.DataFrame({
        "history share": p.mean(),  # the average user's share of the genre in their fit ratings
        f"slot share lambda {REFERENCE}": slots_ref / slots_ref.sum(),
        f"slot share lambda {lam}": slots_lam / slots_lam.sum(),
    })
    table["change (pp)"] = 100 * (table.iloc[:, 2] - table.iloc[:, 1])
    return table.sort_values("change (pp)")


def movers(recs_ref, recs_lam, titles, genres, n=8) -> pd.DataFrame:
    """The n movies that lose the most slots and the n that gain the most."""
    count_ref = recs_ref["item_id"].value_counts()
    count_lam = recs_lam["item_id"].value_counts()
    change = count_lam.sub(count_ref, fill_value=0).astype(int).sort_values()
    picked = pd.concat([change.head(n), change.tail(n)]).index
    return pd.DataFrame({
        "title": titles.loc[picked],
        "slots ref": count_ref.reindex(picked, fill_value=0),
        "slots": count_lam.reindex(picked, fill_value=0),
        "change": change.loc[picked],
        "genres": genres.loc[picked].apply(lambda row: "|".join(row.index[row == 1]), axis=1),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    out = Path(args.out_dir)

    chosen = json.loads((out / "mmr_chosen.json").read_text())["lambda"]
    lambdas = [REFERENCE, chosen, CONTRAST_LAMBDA]

    ratings = load_ratings(args.data_dir)
    genres = genre_vectors(args.data_dir)
    titles = load_movies(args.data_dir).set_index("item_id")["title"]
    fit, target = fit_and_target(temporal_split(ratings), "test")
    users, items = evaluated_users(fit, target), kept_items(fit)
    user_fit = fit[fit["user_id"].isin(users)]
    p = genre_distribution(user_fit, genres)  # each user's genre distribution in the fit ratings

    pool = candidates(fit, items, users)  # the same pool for every lambda
    recs = {lam: mmr_rerank(pool, genres, lam=lam) for lam in lambdas}
    scores = {lam: per_user_scores(recs[lam], target, users, p, genres) for lam in lambdas}

    # The per-user versions must reproduce the team's functions in new_metrics.py.
    for lam in lambdas:
        assert np.isclose(scores[lam]["ILD"].mean(), intra_list_diversity(recs[lam], genres))
        assert np.isclose(scores[lam]["KL"].mean(), calibration_error(recs[lam], fit, genres))

    # 1) Users
    activity = user_fit.groupby("user_id").size().reindex(users)
    entropy = genre_entropy(p).reindex(users)
    groups = {
        "all": pd.Series("all users", index=users),
        f"activity (median {activity.median():.0f} fit ratings)":
            pd.Series(np.where(activity <= activity.median(), "low activity", "high activity"), index=users),
        f"taste breadth (median genre entropy {entropy.median():.2f} nats)":
            pd.Series(np.where(entropy <= entropy.median(), "narrow taste", "broad taste"), index=users),
    }
    users_table = pd.DataFrame([row for lam in (chosen, CONTRAST_LAMBDA)
                                for row in group_rows(scores, groups, lam, args.resamples)])
    users_table.to_csv(out / "who_pays_users_test.csv", index=False)

    # 2) Items
    counts = fit["item_id"].value_counts().reindex(items, fill_value=0).sort_values(ascending=False)
    n_head = int(np.ceil(HEAD_SHARE * len(items)))
    head = set(counts.index[:n_head])
    head_rating_share = counts.iloc[:n_head].sum() / counts.sum()
    items_table = pd.DataFrame([{"lambda": lam, "head items": n_head,
                                 "head share of fit ratings": head_rating_share,
                                 **item_summary(recs[lam], counts, head)} for lam in lambdas])
    items_table.to_csv(out / "who_pays_items_test.csv", index=False)
    for lam in (chosen, CONTRAST_LAMBDA):
        genre_slots(recs[REFERENCE], recs[lam], lam, p, genres).to_csv(out / f"who_pays_genres_test_lam{lam}.csv")
        movers(recs[REFERENCE], recs[lam], titles, genres).to_csv(out / f"who_pays_movers_test_lam{lam}.csv")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"test: {len(users)} users; chosen lambda {chosen} (from mmr_chosen.json), contrast {CONTRAST_LAMBDA}")
    print(users_table[["lambda", "group", "users", "NDCG ref", "NDCG", "dNDCG", "dNDCG low", "dNDCG high",
                       "dKL", "users KL worse %"]].to_string(index=False, float_format="%.4f"))
    print()
    print(items_table.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
