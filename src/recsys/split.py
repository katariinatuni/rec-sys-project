"""Global temporal split (report section 1.3) and the item filter (section 1.2).

Ratings before 1 Jan 2001 00:00 UTC are training data, ratings from then until
1 Jul 2001 00:00 UTC are validation data, and everything later is test data. The first
cut sits where the sign-up wave of 2000 ends (only six users join later), so validation
and test both contain returning users; split_stats.py computes the numbers behind this.

Validation is scored with models fitted on train, and test with models refitted on
train + validation: `fit_and_target` returns the right pair for each period.
"""
from dataclasses import dataclass

import pandas as pd


def unix_seconds(date: str) -> int:
    """Midnight UTC at the start of an ISO date, in the unit of the timestamps."""
    return int(pd.Timestamp(date, tz="UTC").timestamp())


VAL_START = unix_seconds("2001-01-01")
TEST_START = unix_seconds("2001-07-01")
RELEVANT_MIN_RATING = 4  # a held-out rating of 4 or 5 stars counts as relevant
MIN_ITEM_RATINGS = 5     # items with fewer ratings in the fitting data are never recommended


@dataclass(frozen=True)
class Split:
    """The three periods. Each keeps the original row index of the ratings."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_val(self) -> pd.DataFrame:
        """Fitting data for the final test run: train and validation together."""
        return pd.concat([self.train, self.val])


def temporal_split(ratings: pd.DataFrame, val_start: int = VAL_START,
                   test_start: int = TEST_START) -> Split:
    """Cut all ratings at two moments in time (unix seconds).

    A rating stamped exactly at a cut goes to the later period (strict `<`). Every rating
    on one submitted page shares a timestamp, so a page always lands in one period.
    """
    if not val_start < test_start:
        raise ValueError("the validation cut must come before the test cut")
    t = ratings["timestamp"]
    return Split(train=ratings[t < val_start],
                 val=ratings[(t >= val_start) & (t < test_start)],
                 test=ratings[t >= test_start])


def fit_and_target(split: Split, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ratings to fit on, ratings to score on) for "val" or "test"."""
    if period == "val":
        return split.train, split.val
    if period == "test":
        return split.train_val, split.test
    raise ValueError(f"period must be 'val' or 'test', not {period!r}")


def evaluated_users(fit: pd.DataFrame, target: pd.DataFrame,
                    min_rating: int = RELEVANT_MIN_RATING) -> pd.Index:
    """Users scored in a period: at least one earlier rating and one relevant held-out rating.

    A user with no earlier rating gives item-kNN nothing to start from (L1.2 slide 12),
    and a user with no relevant rating has nothing to find, so precision and recall
    would be undefined or zero for reasons unrelated to the model.
    """
    relevant = pd.Index(target.loc[target["rating"] >= min_rating, "user_id"].unique())
    return relevant.intersection(pd.Index(fit["user_id"].unique())).sort_values()


def kept_items(fit: pd.DataFrame, min_ratings: int = MIN_ITEM_RATINGS) -> pd.Index:
    """Items with at least `min_ratings` ratings in the fitting data: the candidate set.

    Counted on the fitting data only, never on all ratings, so no count uses the future.
    """
    counts = fit["item_id"].value_counts()
    return counts.index[counts >= min_ratings].sort_values()
