"""Part 1 data characterization (report section 1.1).

Writes everything to one folder (default output/): the numbers to output/data_stats.json
and the figures to output/figures/. The JSON follows
the report: first what the brief requires (counts, density, time span, per-user and per-item
distributions, ratings), then the findings 1.1 highlights (long tail, sign-up spikes, ratings
concentrated at each user's start, and same-second rating batches).

Run from the project root:
    PYTHONPATH=src python -m recsys.explore --data-dir ../data/ml-1m
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # only write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.data import load_movies, load_ratings

# Each entity keeps one colour in every figure (validated categorical slots).
ITEM_COLOR = "#2a78d6"
USER_COLOR = "#eb6834"
RATING_COLOR = "#6250d6"
NEW_USER_COLOR = "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left",
    "axes.labelcolor": TEXT, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

HOUR, DAY, WEEK = 60 * 60, 24 * 60 * 60, 7 * 24 * 60 * 60  # in seconds, like the timestamps
SESSION_GAP = 10 * 60  # a longer pause between two submissions counts as a new session


def basic_stats(ratings: pd.DataFrame, movies: pd.DataFrame,
                per_user: pd.Series, per_item: pd.Series) -> dict:
    """Counts, density, time span, ratings per user and per item, and the star distribution."""
    t = pd.to_datetime(ratings["timestamp"], unit="s")
    shares = ratings["rating"].value_counts(normalize=True).sort_index()
    return {
        "n_ratings": len(ratings),
        "n_users": len(per_user),
        "n_items_rated": len(per_item),
        "n_items_in_movies_dat": len(movies),  # 177 listed films have no ratings
        # Density over the films that have at least one rating.
        "density": round(len(ratings) / (len(per_user) * len(per_item)), 4),
        "first_rating": str(t.min().date()),
        "last_rating": str(t.max().date()),
        # min is 20 for users: GroupLens kept only users with at least 20 ratings.
        "ratings_per_user": per_user.describe().round(1).to_dict(),
        "ratings_per_item": per_item.describe().round(1).to_dict(),
        "rating_share": {int(k): round(float(v), 4) for k, v in shares.items()},
        "share_4_or_more": round(float((ratings["rating"] >= 4).mean()), 4),
    }


def long_tail_stats(counts: np.ndarray) -> dict:
    """How concentrated the ratings are, given one rating count per item (or per user).

    Shares of all ratings held by the most-rated 1%, 10% and 20% and by the least-rated 50%,
    plus the Gini coefficient (0 = perfectly equal, 1 = one item has everything).
    """
    c = np.sort(np.asarray(counts, dtype=float))[::-1]  # largest count first
    total = c.sum()
    stats = {}
    for pct in (1, 10, 20):
        n_top = max(1, round(len(c) * pct / 100))  # 1% of 3,706 items -> 37 items
        stats[f"top_{pct}pct_share"] = round(float(c[:n_top].sum() / total), 4)
    n_bottom = round(len(c) * 0.5)
    stats["bottom_50pct_share"] = round(float(c[len(c) - n_bottom:].sum() / total), 4)
    x, y = lorenz_curve(c)  # lorenz_curve re-sorts ascending itself
    stats["gini"] = round(float(1 - 2 * np.trapezoid(y, x)), 4)
    return stats


def lorenz_curve(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = share of items (least rated first), y = share of ratings they hold."""
    c = np.sort(np.asarray(counts, dtype=float))
    y = np.concatenate([[0.0], np.cumsum(c) / c.sum()])
    return np.linspace(0, 1, len(y)), y


# --- Activity over time and the sign-up spikes ------------------------------------------

def seconds_since_first_rating(ratings: pd.DataFrame) -> pd.Series:
    """Seconds between each rating and the same user's first rating (0 for that first one).

    Measured from each user's own first rating, so every user gets the same windows
    whatever weekday they joined on. Aligned with `ratings`.
    """
    # transform("min") repeats each user's first timestamp on every one of their rows.
    first = ratings.groupby("user_id")["timestamp"].transform("min")
    return ratings["timestamp"] - first


def weekly_activity(week: pd.Series, is_new: pd.Series) -> pd.DataFrame:
    """Ratings per week (indexed by the Monday it starts), split into "new" and "returning".

    "New" means the rating falls in the user's first week (their sign-up batch).
    """
    counts = pd.crosstab(week, is_new)
    counts = counts.reindex(pd.period_range(week.min(), week.max(), freq="W"), fill_value=0)
    return pd.DataFrame({"new": counts[True].to_numpy(), "returning": counts[False].to_numpy()},
                        index=counts.index.start_time)


def activity_stats(ratings: pd.DataFrame, week: pd.Series, weekly: pd.DataFrame) -> dict:
    """Burstiness, the spike weeks, who made them, and whether particular films drove them."""
    t = pd.to_datetime(ratings["timestamp"], unit="s")
    joined = pd.to_datetime(ratings.groupby("user_id")["timestamp"].min(), unit="s")
    total = weekly["new"] + weekly["returning"]
    # A spike is a week busier than both neighbours that holds at least 2% of all ratings.
    is_peak = (total > total.shift(1)) & (total > total.shift(-1)) & (total >= 0.02 * total.sum())
    spikes = total[is_peak].index

    # Film mix of each spike week compared with the other weeks of 2000. A correlation near 1
    # means the spike rated the same films as usual, so no particular film drove it.
    week_start = week.dt.start_time
    baseline = ratings.loc[(week_start < "2001-01-01") & ~week_start.isin(spikes), "item_id"]
    baseline_share = baseline.value_counts(normalize=True)
    spike_weeks = {}
    for start in spikes:
        share = ratings.loc[week_start == start, "item_id"].value_counts(normalize=True)
        both = pd.concat([share, baseline_share], axis=1).fillna(0)
        spike_weeks[str(start.date())] = {
            "ratings": int(total[start]),
            "share_from_users_in_first_week": round(float(weekly.loc[start, "new"] / total[start]), 3),
            "film_mix_correlation_with_other_2000_weeks": round(float(both.corr().iloc[0, 1]), 3),
        }

    return {
        # Burstiness: days from the first rating until this share of all ratings exists.
        "days_until_share": {f"{q:.0%}": (t.quantile(q) - t.min()).days for q in (0.5, 0.8, 0.9)},
        "spike_weeks": spike_weeks,
        # The README says every user joined in 2000; a handful only rated their first film later.
        "users_first_rating_after_2000": int((joined >= "2001-01-01").sum()),
    }


def user_start_stats(ratings: pd.DataFrame, since_first: pd.Series) -> dict:
    """How strongly each user's ratings concentrate at the start of their activity."""
    first_hour = (since_first < HOUR).groupby(ratings["user_id"]).sum()
    only_first_week = (since_first < WEEK).groupby(ratings["user_id"]).all()
    return {
        "share_of_ratings_within": {label: round(float((since_first < seconds).mean()), 4)
                                    for label, seconds in [("1 hour", HOUR), ("24 hours", DAY),
                                                           ("7 days", WEEK)]},
        "first_hour_ratings_per_user": {"mean": round(float(first_hour.mean()), 1),
                                        "median": float(first_hour.median())},
        # Matches the 20-rating minimum, which suggests it comes from the sign-up step.
        "users_with_20_in_first_hour": int((first_hour >= 20).sum()),
        # Users who never rate again after their first week, so no later split can test them.
        "users_only_active_in_first_week": int(only_first_week.sum()),
    }


# --- Same-second rating batches ---------------------------------------------------------

def batch_stats(ratings: pd.DataFrame, since_first: pd.Series) -> dict:
    """Ratings one user submitted in the same second: pages of the rating interface.

    Batch counts halve from each size to the next up to 10 and then drop about sixfold,
    which points to pages of about 10 films; every rating on a submitted page gets the same
    timestamp, so the order of ratings within a page is unknown.
    """
    groups = ratings.groupby(["user_id", "timestamp"]).agg(size=("rating", "size"),
                                                           stars=("rating", "sum"))
    same_second = ratings.duplicated(["user_id", "timestamp"], keep=False)
    counts = groups["size"].value_counts().sort_index()

    # Stars per batch size, weighted by ratings; 11 stands for "11 or more".
    size_band = groups["size"].clip(upper=11)
    stars_by_size = groups["stars"].groupby(size_band).sum() / groups["size"].groupby(size_band).sum()

    # Time on a page = seconds since the user's previous submission, within one session.
    gaps = groups.reset_index().groupby("user_id")["timestamp"].diff()
    pages = groups.reset_index().assign(gap=gaps.to_numpy(), band=size_band.to_numpy())
    pages = pages[pages["gap"] <= SESSION_GAP]  # also drops each user's first submission (no gap)
    time_by_size = pages.groupby("band")["gap"].quantile([0.25, 0.5, 0.75]).unstack()
    pages_per_size = pages.groupby("band").size()
    # Straight line through the medians for single pages (1-10 ratings):
    # time on page = fixed seconds (reading the page) + seconds per rated film.
    medians = pages[pages["size"] <= 10].groupby("size")["gap"].median()
    per_rating, fixed = np.polyfit(medians.index, medians.to_numpy(), 1)

    def label(size: int) -> str:
        return "11+" if size == 11 else str(int(size))

    return {
        "share_of_ratings_in_batches": round(float(same_second.mean()), 4),
        "share_of_first_hour_ratings_in_batches": round(float(same_second[since_first < HOUR].mean()), 4),
        "batches_by_size": {int(k): int(v) for k, v in counts.items()},
        # How many times more batches of size k there are than of size k+1: about 2 up to 10.
        "count_ratio_to_next_size": {int(k): round(float(counts[k] / counts[k + 1]), 1)
                                     for k in range(1, 12)},
        "mean_stars_by_batch_size": {label(k): round(float(v), 2) for k, v in stars_by_size.items()},
        "share_of_gaps_within_session": round(float(len(pages) / gaps.notna().sum()), 4),
        "page_time_seconds_by_batch_size": {
            label(k): {"batches": int(pages_per_size[k]), "p25": float(row[0.25]),
                       "median": float(row[0.5]), "p75": float(row[0.75])}
            for k, row in time_by_size.iterrows()
        },
        "page_time_fit": {"fixed_seconds": round(float(fixed), 1),
                          "seconds_per_rated_film": round(float(per_rating), 1),
                          "fitted_on": "median time on page for 1-10 ratings"},
    }


# --- Figures ----------------------------------------------------------------------------

def plot_activity_over_time(weekly: pd.DataFrame, n_late_users: int, path: Path) -> None:
    """Ratings per week split by new/returning users, and the cumulative share over time."""
    total = weekly["new"] + weekly["returning"]
    cumulative = total.cumsum() / total.sum()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
    top.stackplot(weekly.index, weekly["new"], weekly["returning"],
                  colors=[NEW_USER_COLOR, RATING_COLOR], linewidth=0,  # an outline would hide the 1-week-wide peaks
                  labels=["users in their first week", "returning users"])
    # After 2000 almost every rating comes from a returning user.
    end_2000 = pd.Timestamp("2001-01-01")
    top.axvline(end_2000, color=MUTED, lw=1, ls=":")
    top.annotate(f"end of 2000: only {n_late_users} users start rating later",
                 (end_2000, total.max() * 0.85), xytext=(6, 0), textcoords="offset points",
                 color=MUTED, fontsize=8)
    top.set(title="Ratings per week", ylabel="ratings")
    top.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    top.legend(frameon=False, fontsize=8, loc="center right")
    bottom.plot(cumulative.index, cumulative.to_numpy(), color=RATING_COLOR, lw=2)
    # Label the reference lines on the left, where the curve is still far below them.
    for level in (0.8, 0.9):
        bottom.axhline(level, color=MUTED, lw=1, ls=":")
        bottom.annotate(f"{level:.0%} of ratings", (cumulative.index[0], level), xytext=(0, 2),
                        textcoords="offset points", color=MUTED, fontsize=8)
    bottom.set(title="Cumulative share of all ratings", ylabel="share", ylim=(0, 1.02))
    bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.savefig(path)
    plt.close(fig)


def plot_rank_frequency(per_item: pd.Series, per_user: pd.Series, path: Path) -> None:
    """Ratings per item and per user, sorted, on log-log axes: the long tail."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    panels = [(axes[0], per_item, ITEM_COLOR, "Ratings per item", "item rank"),
              (axes[1], per_user, USER_COLOR, "Ratings per user", "user rank")]
    for ax, counts, color, title, xlabel in panels:
        values = np.sort(counts.to_numpy())[::-1]
        ax.loglog(np.arange(1, len(values) + 1), values, color=color, lw=2)
        ax.set(title=title, xlabel=f"{xlabel} (most ratings first)", ylabel="ratings")
    # Every user has at least 20 ratings because GroupLens filtered them that way.
    axes[1].axhline(20, color=MUTED, lw=1, ls=":")
    axes[1].annotate("20-rating floor (GroupLens)", (1.5, 20), xytext=(0, 4),
                     textcoords="offset points", color=MUTED, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_rating_distribution(ratings: pd.DataFrame, path: Path) -> None:
    """Share of ratings at each star value, labelled on the bars."""
    shares = ratings["rating"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bars = ax.bar(shares.index, shares.to_numpy(), width=0.7, color=RATING_COLOR,
                  edgecolor="white", linewidth=2)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in shares], fontsize=8, color=TEXT, padding=2)
    ax.set(title="Rating distribution", xlabel="stars", ylabel="share of ratings")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_page_time(batches: dict, path: Path) -> None:
    """Time on a page against how many films were rated on it, with the fixed + per-film fit."""
    timing, fit = batches["page_time_seconds_by_batch_size"], batches["page_time_fit"]
    labels = list(timing)  # "1" ... "10", "11+"
    x = np.arange(len(labels))
    p25, med, p75 = (np.array([timing[k][q] for k in labels]) for q in ("p25", "median", "p75"))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(x, p75 - p25, bottom=p25, width=0.6, color=RATING_COLOR, alpha=0.18, label="middle 50%")
    ax.plot(x, med, color=RATING_COLOR, lw=2, marker="o", ms=4, label="median")
    single = x[:10]  # the fit only covers single pages (1-10 ratings)
    ax.plot(single, fit["fixed_seconds"] + fit["seconds_per_rated_film"] * (single + 1),
            color=MUTED, lw=1, ls="--", label="fit")
    ax.annotate(f"≈ {fit['fixed_seconds']:.0f} s + {fit['seconds_per_rated_film']:.1f} s per film",
                (single[4], fit["fixed_seconds"] + fit["seconds_per_rated_film"] * 5), xytext=(-60, 22),
                textcoords="offset points", color=MUTED, fontsize=8)
    ax.annotate("≈ two pages", (x[-1], med[-1]), xytext=(-62, 4), textcoords="offset points",
                color=TEXT, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set(title="Time on a page", xlabel="films rated on the page (same second)",
           ylabel="seconds since previous submission", ylim=(0, None))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def plot_stars_by_batch_size(batches: dict, overall_mean: float, path: Path) -> None:
    """Mean stars by how many films were rated in the same second; a full page of 10 stands out."""
    stars = batches["mean_stars_by_batch_size"]
    labels = list(stars)
    x = np.arange(len(labels))
    y = np.array([stars[k] for k in labels])
    full = labels.index("10")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.axhline(overall_mean, color=MUTED, lw=1, ls=":")
    # Label the reference line at the right, where the dots have already dropped below it.
    ax.annotate(f"all ratings {overall_mean:.2f}", (x[-1], overall_mean), xytext=(0, 4),
                textcoords="offset points", ha="right", color=MUTED, fontsize=8)
    colors = [RATING_COLOR if i == full else MUTED for i in x]
    ax.scatter(x, y, s=[70 if i == full else 30 for i in x], c=colors,
               edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(f"full page of 10: {y[full]:.2f}", (x[full], y[full]), xytext=(-92, -4),
                textcoords="offset points", color=TEXT, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set(title="Mean stars by batch size", xlabel="films rated in the same second",
           ylabel="mean stars", ylim=(3.0, 3.75))
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


# --- Entry point ------------------------------------------------------------------------

def run(data_dir: str | Path, out_dir: str | Path = "output") -> dict:
    """Load the data, write out_dir/data_stats.json and the figures to out_dir/figures/."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)  # also creates out_dir

    ratings, movies = load_ratings(data_dir), load_movies(data_dir)
    per_user = ratings.groupby("user_id").size()
    per_item = ratings.groupby("item_id").size()
    since_first = seconds_since_first_rating(ratings)
    week = pd.to_datetime(ratings["timestamp"], unit="s").dt.to_period("W")
    weekly = weekly_activity(week, is_new=since_first < WEEK)

    stats = {
        "data": basic_stats(ratings, movies, per_user, per_item),
        "long_tail": {"items": long_tail_stats(per_item.to_numpy()),
                      "users": long_tail_stats(per_user.to_numpy())},
        "activity": activity_stats(ratings, week, weekly),
        "user_start": user_start_stats(ratings, since_first),
        "batches": batch_stats(ratings, since_first),
    }

    plot_activity_over_time(weekly, stats["activity"]["users_first_rating_after_2000"],
                            fig_dir / "activity_over_time.png")
    plot_rank_frequency(per_item, per_user, fig_dir / "long_tail_rank_frequency.png")
    plot_rating_distribution(ratings, fig_dir / "rating_distribution.png")
    plot_page_time(stats["batches"], fig_dir / "page_time_by_batch_size.png")
    plot_stars_by_batch_size(stats["batches"], float(ratings["rating"].mean()),
                             fig_dir / "stars_by_batch_size.png")

    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="folder with ratings.dat and movies.dat")
    parser.add_argument("--out-dir", default="output",
                        help="folder for data_stats.json; figures go to its figures/ subfolder")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
