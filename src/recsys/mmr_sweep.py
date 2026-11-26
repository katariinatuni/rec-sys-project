"""Part 3 (Track A): sweep MMR's lambda, choose it on validation, report every lambda on test.

For each lambda in 0, 0.1, ..., 1.0, MMR (mmr.py) re-ranks the same candidate pool: the 100 best
movies per user from the weighted hybrid (alpha = 0.4). Each period builds its pool once (about
45 s), and every lambda then takes about a second.

1) Validation: each lambda is compared with the lambda that has the best NDCG@10 (paired bootstrap
   over users, seed 0). Selection rule, as in Parts 1-2: among the lambdas that are not clearly
   worse (the 95% interval of the NDCG@10 difference includes zero), take the one with the highest
   intra-list diversity. The choice is written to mmr_chosen.json.
2) Test: the same sweep, with each lambda compared with lambda = 1 (the hybrid's own top-10).

Run from the repository root:
    PYTHONPATH=src python -m recsys.mmr_sweep --data-dir ../data/ml-1m --out-dir output_part3
Options: --resamples N (default 5000) and --out-dir (default output). Writes mmr_sweep_val.csv,
mmr_chosen.json, mmr_sweep_test.csv, part3A_diversity_rows.csv and figures/mmr_frontier.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write the figure to a file, no window
import matplotlib.pyplot as plt
import pandas as pd

from recsys.compare import bootstrap_diff
from recsys.data import load_ratings
from recsys.metrics import catalogue_coverage, per_user_metrics, ranked_lists
from recsys.mmr import HYBRID_ALPHA, N_CANDIDATES, candidates, genre_vectors, mmr_rerank
from recsys.new_metrics import calibration_error, gini, intra_list_diversity
from recsys.split import evaluated_users, fit_and_target, kept_items, temporal_split

LAMBDAS = [round(0.1 * step, 1) for step in range(11)]  # 0.0, 0.1, ..., 1.0


def sweep(ratings, genres, period):
    """Every lambda on one period: a table with one row of metrics per lambda, and each lambda's
    per-user NDCG@10 (same user order for every lambda, which the paired bootstrap needs)."""
    fit, target = fit_and_target(temporal_split(ratings), period)
    users, items = evaluated_users(fit, target), kept_items(fit)
    pool = candidates(fit, items, users)  # built once; every lambda re-ranks the same pool

    rows, ndcg_per_user = [], []
    for lam in LAMBDAS:
        recs = mmr_rerank(pool, genres, lam=lam)
        accuracy = per_user_metrics(recs, target, users, k=10)
        rows.append({
            "lambda": lam,
            **accuracy.mean().to_dict(),  # Precision@10, Recall@10, NDCG@10
            "Coverage": catalogue_coverage(list(ranked_lists(recs).values()), items, 10),
            "ILD": intra_list_diversity(recs, genres),
            "Gini": gini(recs, items),
            "Calibration error": calibration_error(recs, fit, genres),
        })
        ndcg_per_user.append(accuracy["NDCG@10"].to_numpy())
    print(f"{period}: {len(users)} users, {len(LAMBDAS)} lambdas")
    return pd.DataFrame(rows), ndcg_per_user


def add_difference(table, ndcg_per_user, reference, name, resamples):
    """Each lambda's NDCG@10 minus the reference lambda's, with its paired 95% bootstrap interval."""
    reference_scores = ndcg_per_user[LAMBDAS.index(reference)]
    diffs = [bootstrap_diff(scores, reference_scores, resamples) for scores in ndcg_per_user]
    table[f"NDCG vs {name}"] = [d[0] for d in diffs]
    table["CI low"] = [d[1] for d in diffs]
    table["CI high"] = [d[2] for d in diffs]
    table["clearly worse"] = table["CI high"] < 0  # the whole interval is below zero
    return table


def choose(table) -> float:
    """Not clearly worse in NDCG@10 than the best lambda, then the highest intra-list diversity."""
    eligible = table[~table["clearly worse"]]
    return float(eligible.loc[eligible["ILD"].idxmax(), "lambda"])


def diversity_rows(test, chosen) -> pd.DataFrame:
    """The test sweep in the columns of the results template's "Part 3A - Diversity" sheet."""
    notes = []
    for lam, diff, low, high in zip(test["lambda"], test["NDCG vs lambda 1"], test["CI low"], test["CI high"]):
        if lam == 1.0:
            notes.append("unmodified base recommender (reference)")
        else:
            chosen_note = "chosen on validation; " if lam == chosen else ""
            notes.append(f"{chosen_note}NDCG@10 vs lambda 1 = {diff:+.4f} [{low:+.4f}, {high:+.4f}]")
    return pd.DataFrame({
        "Base model": f"Weighted hybrid (alpha = {HYBRID_ALPHA}), top-{N_CANDIDATES} candidates",
        "Re-ranking method": "MMR (greedy, lambda * relevance - (1 - lambda) * max similarity)",
        "Trade-off parameter": [f"lambda = {lam:.1f}" for lam in test["lambda"]],
        "Item distance": "1 - cosine similarity of 18-dim binary genre vectors",
        "Precision@10": test["Precision@10"],
        "Recall@10": test["Recall@10"],
        "NDCG@10": test["NDCG@10"],
        "Intra-list diversity": test["ILD"],
        "Catalogue coverage": test["Coverage"],
        "Gini": test["Gini"],
        "Calibration error": test["Calibration error"],
        "Notes": notes,
    })


def frontier_figure(val, test, chosen, path) -> None:
    """NDCG@10 against ILD, calibration error and coverage: one point per lambda, both periods."""
    blue, orange = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [("ILD", "Intra-list diversity (genre, 1 - cosine)"),
              ("Calibration error", "Calibration error KL(p || q~), lower = better"),
              ("Coverage", "Catalogue coverage")]
    for ax, (column, label) in zip(axes, panels):
        for table, colour, name in [(val, orange, "validation"), (test, blue, "test")]:
            ax.plot(table[column], table["NDCG@10"], "-o", color=colour, lw=2, ms=5, label=name)
        # Label a few lambdas on the test curve only, so the labels never overlap.
        for _, row in test[test["lambda"].isin([0.0, 0.5, 0.7, 0.8, 1.0])].iterrows():
            ax.annotate(f"{row['lambda']:.1f}", (row[column], row["NDCG@10"]), fontsize=7,
                        color="#555", xytext=(4, 3), textcoords="offset points")
        pick = test[test["lambda"] == chosen].iloc[0]
        ax.scatter([pick[column]], [pick["NDCG@10"]], s=140, facecolors="none",
                   edgecolors="#222", lw=1.2, zorder=5)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("NDCG@10")
    axes[0].legend(frameon=False, fontsize=8, title="period (labels = λ on test)", title_fontsize=8)
    fig.suptitle(f"MMR over the weighted hybrid (α = {HYBRID_ALPHA}): accuracy–diversity frontier; "
                 f"ring = λ = {chosen}, chosen on validation", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(args.data_dir)
    genres = genre_vectors(args.data_dir)

    # 1) Validation: compare with the best lambda, then choose.
    val, val_ndcg = sweep(ratings, genres, "val")
    best = float(val.loc[val["NDCG@10"].idxmax(), "lambda"])
    val = add_difference(val, val_ndcg, best, "best", args.resamples)
    chosen = choose(val)
    val.to_csv(out / "mmr_sweep_val.csv", index=False)
    (out / "mmr_chosen.json").write_text(json.dumps({"lambda": chosen}, indent=2) + "\n")
    print(val.to_string(index=False, float_format="%.4f"))
    print(f"\nbest NDCG@10 at lambda {best}; chosen: lambda {chosen} "
          f"({(~val['clearly worse']).sum()} lambdas not clearly worse than the best)\n")

    # 2) Test: the same sweep, every lambda against the unmodified hybrid.
    test, test_ndcg = sweep(ratings, genres, "test")
    test = add_difference(test, test_ndcg, 1.0, "lambda 1", args.resamples)
    test.to_csv(out / "mmr_sweep_test.csv", index=False)
    diversity_rows(test, chosen).to_csv(out / "part3A_diversity_rows.csv", index=False)
    frontier_figure(val, test, chosen, out / "figures" / "mmr_frontier.png")
    print(test.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
