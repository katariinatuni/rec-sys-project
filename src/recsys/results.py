"""Write output/results.csv in the columns of the course results template ("Part 1 - Baselines").

Reads the two tables compare.py saves (compare_val.csv and compare_test.csv), chooses one
variant per model on validation, and writes every variant for both periods. Nothing is
recomputed here, so results.csv always agrees with compare.py.

Selection rule, on validation only: among a model's variants that are not clearly worse than
the best variant in the comparison (the 95% paired-bootstrap interval of the NDCG@10
difference includes zero), take the one with the highest catalogue coverage. If none of a
model's variants qualifies, take its highest NDCG@10. Accuracy and coverage are reported as a
pair (Lecture 3.1, slide 9).

Run from the repository root, after compare.py has run on both periods (under a second):
    PYTHONPATH=src python -m recsys.results
"""
import argparse
from pathlib import Path

import pandas as pd

SHEET = "Part 1 - Baselines"
# The template's required columns, in its order. The format is mandatory: do not rename.
TEMPLATE_COLUMNS = ["Model", "Variant / settings", "Similarity", "k", "Precision@10",
                    "Recall@10", "NDCG@10", "Catalogue coverage", "Seed", "Notes"]
MODEL_NAMES = {"Item kNN": "Item-based kNN", "Popular": "Most popular"}
# compare.py's similarity name -> (what the model learns from, similarity as written in the template)
KNN_SIMILARITIES = {
    "adjusted_cosine": ("ratings minus each user's mean; sum over the item's k neighbours",
                        "adjusted cosine"),
    "cosine": ("raw ratings; sum over the item's k neighbours", "cosine"),
}
VARIANT = ["model", "setting", "counting"]  # together these name one variant in compare.py


def choose(val: pd.DataFrame) -> list[tuple]:
    """The chosen variant of each model, by the selection rule in the module docstring."""
    chosen = []
    for _, rows in val.groupby("model", sort=False):
        qualifying = rows[~rows["clearly worse"]]
        if len(qualifying):
            pick = qualifying.sort_values(["Coverage", "NDCG@10"], ascending=False, kind="stable")
        else:
            pick = rows.sort_values("NDCG@10", ascending=False, kind="stable")
        chosen.append(tuple(pick.iloc[0][VARIANT]))
    return chosen


def template_rows(table: pd.DataFrame, period: str, chosen: list[tuple]) -> list[dict]:
    """One row per variant: the template's columns, plus the period and the comparison."""
    rows = []
    for r in table.to_dict("records"):
        if r["model"] == "Item kNN":
            similarity, k = r["setting"].split(", k = ")  # compare.py writes "cosine, k = 30"
            settings, similarity = KNN_SIMILARITIES[similarity]
            k = int(k)
        else:
            settings, similarity, k = f"{r['setting']}, {r['counting']}", "-", "-"
        rows.append({
            "Sheet": SHEET, "Period": period,
            "Model": MODEL_NAMES[r["model"]], "Variant / settings": settings,
            "Similarity": similarity, "k": k,
            "Precision@10": r["Precision@10"], "Recall@10": r["Recall@10"],
            "NDCG@10": r["NDCG@10"], "Catalogue coverage": r["Coverage"],
            "Seed": "none (deterministic); bootstrap seed 0",
            "Notes": "chosen on validation" if tuple(r[c] for c in VARIANT) in chosen else "",
            # Extra columns (the template allows adding them after the required ones):
            "NDCG@10 vs best in period": r["NDCG vs best"],
            "CI low": r["CI low"], "CI high": r["CI high"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="output",
                        help="folder with compare_val.csv and compare_test.csv; results.csv goes here")
    args = parser.parse_args()
    out = Path(args.out_dir)

    val = pd.read_csv(out / "compare_val.csv")
    test = pd.read_csv(out / "compare_test.csv")
    if not val[VARIANT].equals(test[VARIANT]):
        raise ValueError("compare_val.csv and compare_test.csv list different variants")

    chosen = choose(val)  # validation only: test plays no part in the choice
    rows = template_rows(val, "validation", chosen) + template_rows(test, "test", chosen)
    pd.DataFrame(rows).round(4).to_csv(out / "results.csv", index=False)

    print(f"Wrote {out / 'results.csv'} ({len(rows)} rows). Chosen on validation:")
    val_by_variant, test_by_variant = val.set_index(VARIANT), test.set_index(VARIANT)
    for key in chosen:
        v, t = val_by_variant.loc[key], test_by_variant.loc[key]
        print(f"  {MODEL_NAMES[key[0]]:<15} {key[1]}, {key[2]}: validation NDCG@10 {v['NDCG@10']:.4f}"
              f" / coverage {v['Coverage']:.1%}; test NDCG@10 {t['NDCG@10']:.4f}"
              f" / coverage {t['Coverage']:.1%}")


if __name__ == "__main__":
    main()
