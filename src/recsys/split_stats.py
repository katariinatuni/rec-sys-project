"""Every number quoted in report sections 1.2 (preprocessing) and 1.3 (split).

Writes output/split_stats.json, and Table 1 (thresholds) and Table 2 (split) as
output/report_tables.html and output/report_tables.md. The tables are built from the
JSON, so they always match it. To paste a table into Word, open the HTML file in a
browser, select the table and copy it.

Run from the project root:
    PYTHONPATH=src python -m recsys.split_stats --data-dir ../data/ml-1m
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_ratings
from recsys.explore import WEEK, seconds_since_first_rating
from recsys.split import (MIN_ITEM_RATINGS, RELEVANT_MIN_RATING, TEST_START, VAL_START, Split,
                          evaluated_users, fit_and_target, kept_items, temporal_split)


def utc(seconds: float) -> str:
    return pd.Timestamp(int(seconds), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")


# --- Numbers ----------------------------------------------------------------------------

def period_stats(fit: pd.DataFrame | None, target: pd.DataFrame, n_ratings: int) -> dict:
    """One period: its row in Table 2 and, for a scored period, its column in Table 1.

    `fit` is the data models are fitted on before scoring `target`; None for train.
    """
    stats = {
        "first": utc(target["timestamp"].min()),
        "last": utc(target["timestamp"].max()),
        "ratings": len(target),
        "share_of_ratings": round(len(target) / n_ratings, 4),
        "users": int(target["user_id"].nunique()),
    }
    if fit is None:
        return stats

    # Earlier ratings per user of this period (0 for users who only start now).
    earlier = fit["user_id"].value_counts().reindex(target["user_id"].unique(), fill_value=0)
    no_history = earlier.index[earlier == 0]
    evaluated = evaluated_users(fit, target)
    relevant = target["rating"] >= RELEVANT_MIN_RATING
    relevant_per_user = target[relevant & target["user_id"].isin(evaluated)].groupby("user_id").size()
    kept = kept_items(fit)
    n_fit_items = int(fit["item_id"].nunique())
    fit_dropped = ~fit["item_id"].isin(kept)
    unreachable = ~target["item_id"].isin(kept)  # dropped by the filter or never rated before

    return stats | {
        # Table 2
        "users_without_earlier_ratings": len(no_history),
        "evaluated_users": len(evaluated),
        # Table 1, relevance row
        "share_relevant": round(float(relevant.mean()), 4),
        "median_relevant_per_evaluated_user": float(relevant_per_user.median()),
        "users_without_relevant_rating": int(earlier.size - len(no_history) - len(evaluated)),
        # Table 1, history row
        "ratings_of_users_without_earlier_ratings": int(target["user_id"].isin(no_history).sum()),
        # Table 1, item filter and candidate set rows; text: "about 8 % of items"
        "items_in_fit_data": n_fit_items,
        "items_dropped": n_fit_items - len(kept),
        "share_of_items_dropped": round((n_fit_items - len(kept)) / n_fit_items, 4),
        "fit_ratings_on_dropped_items": int(fit_dropped.sum()),
        "share_of_fit_ratings_dropped": round(float(fit_dropped.mean()), 4),
        "held_out_ratings_unreachable": int(unreachable.sum()),
        "relevant_held_out_ratings_unreachable": int((unreachable & relevant).sum()),
        "items_kept": len(kept),
    }


def share_based_split(ratings: pd.DataFrame, cut_shares: tuple[float, float] = (0.8, 0.9)) -> dict:
    """The conventional 80/10/10 split by share of ratings, for comparison with ours."""
    t = np.sort(ratings["timestamp"].to_numpy())
    val_start, test_start = (int(t[int(len(t) * s)]) for s in cut_shares)
    split = temporal_split(ratings, val_start, test_start)
    stats = {"val_start": utc(val_start), "test_start": utc(test_start)}
    for period in ("val", "test"):
        fit, target = fit_and_target(split, period)
        stats[period] = {"ratings": len(target), "evaluated_users": len(evaluated_users(fit, target))}
    return stats


def after_2000_stats(ratings: pd.DataFrame, split: Split, since_first: pd.Series) -> dict:
    """Who rates after 2000 and how, against 2000 (the training period)."""
    after = pd.concat([split.val, split.test])
    stayers = after["user_id"].unique()
    per_user_2000 = split.train["user_id"].value_counts()
    stayed = per_user_2000.index.isin(stayers)
    first_rating = ratings.groupby("user_id")["timestamp"].min()
    test = split.test
    # The top 1 % of films by 2000 ratings: counted on training data only.
    counts_2000 = split.train["item_id"].value_counts()
    head = counts_2000.index[:max(1, round(len(counts_2000) * 0.01))]

    return {
        "users_first_rating_after_2000": int((first_rating >= VAL_START).sum()),
        "train_share_from_users_in_first_week": round(float((since_first.loc[split.train.index] < WEEK).mean()), 4),
        "test_users": int(test["user_id"].nunique()),
        "test_users_also_rating_in_val": int(np.isin(test["user_id"].unique(), split.val["user_id"].unique()).sum()),
        "users_rating_after_2000": len(stayers),
        "share_of_users_rating_after_2000": round(len(stayers) / ratings["user_id"].nunique(), 4),
        "median_ratings_in_2000": {"users_rating_after_2000": float(per_user_2000[stayed].median()),
                                   "other_users": float(per_user_2000[~stayed].median())},
        "test_share_in_same_second_batches": round(float(test.duplicated(["user_id", "timestamp"], keep=False).mean()), 4),
        "top_1pct_films_share_of_ratings": {"2000": round(float(split.train["item_id"].isin(head).mean()), 4),
                                            "test": round(float(test["item_id"].isin(head).mean()), 4)},
    }


# --- Tables -----------------------------------------------------------------------------

Table = tuple[str, list[str], list[list[str]]]  # caption, header, rows


def num(n: int) -> str:
    return f"{n:,}"


def pct(share: float, digits: int = 1) -> str:
    return f"{share * 100:.{digits}f} %"


def date_range(first: str, last: str) -> str:
    """"25 Apr – 31 Dec 2000", or "1 Jul 2001 – 28 Feb 2003" across years."""
    a, b = pd.Timestamp(first), pd.Timestamp(last)
    start = f"{a.day} {a:%b}" if a.year == b.year else f"{a.day} {a:%b %Y}"
    return f"{start} – {b.day} {b:%b %Y}"


def threshold_table(stats: dict) -> Table:
    def per_period(cell) -> list[str]:
        return [cell(stats["periods"][p]) for p in ("val", "test")]

    rows = [
        ["Relevance", f"held-out rating ≥ {RELEVANT_MIN_RATING}", *per_period(lambda p: (
            f"{pct(p['share_relevant'])} of ratings relevant; "
            f"median {p['median_relevant_per_evaluated_user']:.0f} per evaluated user; "
            f"{p['users_without_relevant_rating']} users have none and are not scored"))],
        ["History", "≥ 1 earlier rating to be evaluated", *per_period(lambda p: (
            f"{p['users_without_earlier_ratings']} users "
            f"({num(p['ratings_of_users_without_earlier_ratings'])} ratings) not scored"))],
        ["Item filter", f"≥ {MIN_ITEM_RATINGS} ratings in the fitting data", *per_period(lambda p: (
            f"{p['items_dropped']} of {num(p['items_in_fit_data'])} items dropped; "
            f"{num(p['fit_ratings_on_dropped_items'])} fitting ratings ({pct(p['share_of_fit_ratings_dropped'], 2)}); "
            f"{p['held_out_ratings_unreachable']} held-out ratings unreachable, "
            f"{p['relevant_held_out_ratings_unreachable']} of them relevant"))],
        ["Candidate set", "items kept by the filter", *per_period(lambda p: f"{num(p['items_kept'])} items")],
    ]
    return "Table 1 - Thresholds", ["Threshold", "Rule", "Validation", "Test"], rows


def split_table(stats: dict) -> Table:
    def count(value: int | None) -> str:
        return "–" if value is None else num(value)

    rows = []
    for name, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        p = stats["periods"][name]
        rows.append([label, date_range(p["first"], p["last"]),
                     f"{num(p['ratings'])} ({pct(p['share_of_ratings'])})", num(p["users"]),
                     count(p.get("users_without_earlier_ratings")), count(p.get("evaluated_users"))])
    # Every header cell is named: Word drops an empty first header cell on paste, which
    # shifts all headers one column left.
    header = ["Set", "Period (UTC)", "Ratings", "Users", "Users with no earlier ratings", "Evaluated users"]
    return "Table 2 - Split", header, rows


def markdown(table: Table) -> str:
    caption, header, rows = table
    lines = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def html_table(table: Table) -> str:
    """A table with inline styles, which survive copying from a browser into Word."""
    caption, header, rows = table
    # Right-align columns that hold only counts, like "926", "47,314 (4.7 %)" or "–".
    count = re.compile(r"[\d,]+( \([\d.]+ %\))?|–")
    numeric = [all(count.fullmatch(row[c]) for row in rows) for c in range(len(header))]
    border = "border:1px solid #999;padding:3px 6px;vertical-align:top;"

    def cell(tag: str, text: str, col: int) -> str:
        align = "right" if numeric[col] and tag == "td" else "left"
        return f'<{tag} style="{border}text-align:{align}">{html.escape(text)}</{tag}>'

    head = "".join(cell("th", h, c) for c, h in enumerate(header))
    body = "".join("<tr>" + "".join(cell("td", v, c) for c, v in enumerate(row)) + "</tr>" for row in rows)
    return (f'<p style="font-weight:bold">{html.escape(caption)}</p>\n'
            f'<table style="border-collapse:collapse;font-family:Calibri,Arial,sans-serif;font-size:10pt">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>\n")


def write_tables(stats: dict, out_dir: Path) -> None:
    tables = [threshold_table(stats), split_table(stats)]
    (out_dir / "report_tables.md").write_text("\n".join(markdown(t) for t in tables))
    page = "\n<br>\n".join(html_table(t) for t in tables)
    (out_dir / "report_tables.html").write_text(
        f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>Report tables</title></head>\n'
        f'<body style="font-family:Calibri,Arial,sans-serif;max-width:60rem">\n{page}</body></html>\n')


def run(data_dir: str | Path, out_dir: str | Path = "output") -> dict:
    """Load the data, split it, write out_dir/split_stats.json and the report tables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(data_dir)
    since_first = seconds_since_first_rating(ratings)
    split = temporal_split(ratings)

    stats = {
        "cuts": {"val_start": utc(VAL_START), "test_start": utc(TEST_START)},
        "periods": {"train": period_stats(None, split.train, len(ratings)),
                    **{p: period_stats(*fit_and_target(split, p), len(ratings)) for p in ("val", "test")}},
        "section_1_2": {"min_ratings_per_user": int(ratings.groupby("user_id").size().min())},
        "section_1_3": {"share_80_10_10": share_based_split(ratings),
                        **after_2000_stats(ratings, split, since_first)},
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    write_tables(stats, out_dir)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="folder with ratings.dat")
    parser.add_argument("--out-dir", default="output", help="folder for split_stats.json and the tables")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
