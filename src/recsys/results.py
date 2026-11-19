"""Write output/results.csv in the columns of the course results template ("Part 2 - Models").

Reads the two tables compare.py saves (compare_val.csv and compare_test.csv), chooses one
variant per model on validation, and writes every variant for both periods. Nothing is
recomputed here, so results.csv always agrees with compare.py.

Selection rule, on validation only: among a model's variants that are not clearly worse than
the best variant in the comparison (the 95% paired-bootstrap interval of the NDCG@10
difference includes zero), take the one with the highest catalogue coverage. If none of a
model's variants qualifies, take its highest NDCG@10. Accuracy and coverage are reported as a
pair (Lecture 3.1, slide 9).

Variability: every model is deterministic, so there are no seeds to vary. The template's
"Variability / CI / test" column holds the paired test across users that compare.py ran:
the variant's NDCG@10 difference to the best variant of the period, with its 95% interval.

Run from the repository root, after compare.py has run on both periods (under a second):
    PYTHONPATH=src python -m recsys.results
"""
import argparse
from pathlib import Path

import pandas as pd

SHEET = "Part 2 - Models"  # the template's sheet name: do not rename
# The template's required columns, in its order. The format is mandatory: do not rename.
TEMPLATE_COLUMNS = ["Model", "Paradigm", "Self-implemented?", "Key hyperparameters",
                    "Precision@10", "Recall@10", "NDCG@10", "Catalogue coverage",
                    "Variability / CI / test", "Seed(s)", "Notes"]
# compare.py's model name -> the name, paradigm and self-implemented label written in the template
MODEL_NAMES = {"Item kNN": "Item-based kNN", "User kNN": "User-based kNN", "Markov": "Markov chain",
               "Popular": "Most popular"}
PARADIGMS = {"Item kNN": "Neighbourhood CF (item-based)", "User kNN": "Neighbourhood CF (user-based)",
             "Markov": "Sequential (Markov chain)", "Popular": "Non-personalised"}
SELF_IMPLEMENTED = {"Item kNN": "No (Part 1 baseline)", "User kNN": "Yes", "Markov": "Yes",
                    "Popular": "No (Part 1 baseline)"}
# compare.py's similarity name -> (similarity as written in the template, what the model learns from)
KNN_SIMILARITIES = {
    "adjusted_cosine": ("adjusted cosine", "ratings minus each user's mean; sum over the item's k neighbours"),
    "cosine": ("cosine", "raw ratings; sum over the item's k neighbours"),
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


def hyperparameters(model: str, setting: str, counting: str) -> str:
    """One variant's settings in words, from the names compare.py gives it."""
    if model == "Item kNN":
        similarity, k = setting.split(", k = ")  # compare.py writes "cosine, k = 30"
        similarity, learns_from = KNN_SIMILARITIES[similarity]
        return f"{similarity}, k = {k}; {learns_from}"

    if model == "User kNN":
        rest, k = setting.split(", k = ")  # compare.py writes "pearson, gamma = 30, k = 50"
        gamma = int(rest.split(", gamma = ")[1])  # "pearson, gamma = 30" -> 30
        # gamma = 1 means no shrinking (weight min(n, 1) / 1 = 1), otherwise write out the weight formula
        weighting = "none" if gamma == 1 else f"min(n, {gamma}) / {gamma}"
        return (f"Pearson, k = {k}, significance weighting {weighting} (gamma = {gamma}); "
                "ratings minus each user's mean; sum over the user's k neighbours")

    if model == "Markov":
        window, pop_weight = setting.split(", pop_weight = ")  # compare.py writes "window = 5, pop_weight = 0.3"
        window = int(window.split("window = ")[1])  # "window = 5" -> 5
        # The settings after the first semicolon are fixed: recommend()'s defaults, from markov_tuning.py.
        return (f"window {window}, popularity mix {pop_weight}; pairs >= 1 day apart; "
                "state = whole history, no decay")

    return f"{setting}, {counting}"  # Popular: "14 days, ratings >= 4"


def template_rows(table: pd.DataFrame, period: str, chosen: list[tuple]) -> list[dict]:
    """One row per variant: the template's columns, plus the period and the comparison."""
    # The variant every difference in this period is measured against (compare.py: highest mean NDCG@10).
    best = table.loc[table["NDCG@10"].idxmax()]
    best_name = f"{MODEL_NAMES[best['model']]}, {best['setting']}"

    rows = []
    for r in table.to_dict("records"):
        if r["NDCG vs best"] == 0:
            variability = "best NDCG@10 in this period (the reference of the paired test)"
        else:
            variability = (f"NDCG@10 vs best in period ({best_name}): {r['NDCG vs best']:+.4f}, "
                           f"95% CI [{r['CI low']:+.4f}, {r['CI high']:+.4f}], paired bootstrap over users")
        rows.append({
            "Sheet": SHEET, "Period": period,
            "Model": MODEL_NAMES[r["model"]], "Paradigm": PARADIGMS[r["model"]],
            "Self-implemented?": SELF_IMPLEMENTED[r["model"]],
            "Key hyperparameters": hyperparameters(r["model"], r["setting"], r["counting"]),
            "Precision@10": r["Precision@10"], "Recall@10": r["Recall@10"],
            "NDCG@10": r["NDCG@10"], "Catalogue coverage": r["Coverage"],
            "Variability / CI / test": variability,
            "Seed(s)": "none (deterministic); bootstrap seed 0",
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
