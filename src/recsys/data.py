"""Load the MovieLens 1M ratings, movies and users from the .dat files (Latin-1, "::")."""
from pathlib import Path

import pandas as pd

# The .dat files use a two-character separator ("::"), which only pandas'
# python engine can split on, and Latin-1 encoding (50 titles have accents).
_DAT = dict(sep="::", engine="python", encoding="latin-1", header=None)


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    """One row per rating: user_id, item_id, rating (1-5), timestamp (unix seconds)."""
    return pd.read_csv(Path(data_dir) / "ratings.dat",
                       names=["user_id", "item_id", "rating", "timestamp"], **_DAT)


def load_movies(data_dir: str | Path) -> pd.DataFrame:
    """One row per movie: item_id, title, genres (list of str), year (release year)."""
    movies = pd.read_csv(Path(data_dir) / "movies.dat",
                         names=["item_id", "title", "genres"], **_DAT)
    movies["genres"] = movies["genres"].str.split("|")
    # The release year is the last "(YYYY)" in the title, e.g. "Toy Story (1995)".
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("Int64")
    return movies


def load_users(data_dir: str | Path) -> pd.DataFrame:
    """One row per user: user_id, gender, age (band code), occupation (code), zip."""
    return pd.read_csv(Path(data_dir) / "users.dat",
                       names=["user_id", "gender", "age", "occupation", "zip"],
                       dtype={"zip": str}, **_DAT)
