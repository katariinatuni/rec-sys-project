"""Sparsity ablation: the chosen variant of each model trained on 100%, 50% and 20% of the fit data.

The fit ratings are subsampled at random (seeds 0, 1 and 2). The held-out ratings, the evaluated
users and the candidate movies stay the same at every fraction. A user left with no fit ratings
gets no recommendations and scores 0. 

At every fraction, each model is compared with that fraction's best model by a paired bootstrap
over users (as in compare.py), on each user's NDCG@10 averaged over the seeds.

Run from the repository root:
    PYTHONPATH=src python -m recsys.sparsity_ablation --data-dir ../data/ml-1m
Writes ablation_<period>.csv in the columns of the course template (mean over the seeds),
with the comparison columns after them."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.breakdown import run_variant
from recsys.compare import bootstrap_diff
from recsys.data import load_ratings
from recsys.metrics import evaluate, per_user_metrics
from recsys.results import choose, MODEL_NAMES
from recsys.split import temporal_split, fit_and_target, evaluated_users, kept_items

FRACTIONS = [1.0, 0.5, 0.2]  # share of the fit ratings kept
SEEDS = [0, 1, 2]  # one random subsample per seed
# The template's columns, in its order (its "Notes" column is left out). Do not rename.
TEMPLATE_COLUMNS = ["Model", "Training interactions (%)", "Precision@10", "Recall@10", "NDCG@10",
                    "Catalogue coverage", "Seed(s)"]
# The comparison with the best model of the same fraction, after the template's columns.
COMPARISON_COLUMNS = ["Best model at this fraction", "NDCG@10 vs best model", "CI low", "CI high",
                      "Clearly worse than the best model"]


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
    users = evaluated_users(fit, target)  # the same users at every fraction
    items = kept_items(fit)  # the same candidate movies at every fraction
    chosen = choose(pd.read_csv(out / "compare_val.csv"))  # variants chosen on validation

    rows = []  # one row per model and fraction
    for fraction in FRACTIONS:
        fraction_rows = []  # this fraction's rows, one per model
        fraction_ndcg = []  # for each of them: every user's NDCG@10, averaged over the seeds
        if fraction == 1.0:
            seeds = [0]  # the full data has nothing random: one run is enough
        else:
            seeds = SEEDS

        for model, setting, counting in chosen:
            results = []  # this model's metrics at this fraction, one entry per seed
            ndcg_per_seed = []  # every user's NDCG@10, one array per seed
            for seed in seeds:
                sample = fit.sample(frac=fraction, random_state=seed)  # random share of the fit ratings
                sample = sample.sort_index()  # original row order, so 100% equals the full data
                recs = run_variant(model, setting, counting, sample, items, users)  # train and recommend
                metrics = evaluate(recs, target, users, items, k=10)  # same held-out ratings every time
                results.append(metrics)
                ndcg_per_seed.append(per_user_metrics(recs, target, users, k=10)["NDCG@10"].to_numpy())
                print(f"{fraction:.0%} seed {seed}: {MODEL_NAMES[model]} NDCG@10 {metrics['NDCG@10']:.4f}",
                      flush=True)  # progress; the whole run takes several minutes

            mean = pd.DataFrame(results).mean()  # mean of each metric over the seeds
            fraction_rows.append({"Model": MODEL_NAMES[model],
                                  "Training interactions (%)": round(fraction * 100),  # e.g. 50
                                  "Precision@10": mean["Precision@10"],
                                  "Recall@10": mean["Recall@10"],
                                  "NDCG@10": mean["NDCG@10"],
                                  "Catalogue coverage": mean["Coverage"],
                                  "Seed(s)": ", ".join(str(seed) for seed in seeds)})  # e.g. "0, 1, 2"
            fraction_ndcg.append(np.mean(ndcg_per_seed, axis=0))  # each user's mean over the seeds

        # Compare each model with this fraction's best model (paired bootstrap over users).
        best = np.argmax([row["NDCG@10"] for row in fraction_rows])  # position of the best model
        for i, row in enumerate(fraction_rows):
            # For the best model itself the difference and its interval are 0.
            diff, low, high = bootstrap_diff(fraction_ndcg[i], fraction_ndcg[best], args.resamples)
            row["Best model at this fraction"] = i == best
            row["NDCG@10 vs best model"] = diff
            row["CI low"] = low
            row["CI high"] = high
            row["Clearly worse than the best model"] = high < 0  # the whole interval below zero
        rows.extend(fraction_rows)  # this fraction's rows into the table

    table = pd.DataFrame(rows, columns=TEMPLATE_COLUMNS + COMPARISON_COLUMNS)
    print(table.to_string(index=False, float_format="%.4f"))
    table.round(4).to_csv(out / f"ablation_{args.period}.csv", index=False)


if __name__ == "__main__":
    main()