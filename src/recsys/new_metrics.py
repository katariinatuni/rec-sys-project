"""Part 3 new metrics: intra-list diversity, Gini and calibration error.

All three take a recommendation table with columns user_id, item_id (one row per
recommended movie), like the one mmr.py returns.
Genre vectors come from genre_vectors in mmr.py (one 0/1 column per genre).
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

alfa = 0.01  # small share of the user's own distribution mixed into the list's, so we never divide by zero


def intra_list_diversity(recs, genres) -> float:
    """Mean pairwise genre distance (1 - cosine) inside each list, averaged over users."""
    values = []
    for user, group in recs.groupby("user_id", sort=False): # iterating through every user and their recommendations
        vectors = genres.loc[group["item_id"]].to_numpy() 
        # Retrieve the rows for the movies in the genre table and convert them into a NumPy matrix with 10 rows (movies) and 18 columns (genres).
        if len(vectors) < 2:
            continue  # a list with one movie has no pairs
        distance = 1 - cosine_similarity(vectors)  # a 10 × 10 matrix in which the similarity between each pair of movies ranges from 0 to 1. Subtracting this value from 1 yields the distance. Same genre = 0, no shared genres = 1.
        pairs = np.triu_indices(len(vectors), k=1)  # each pair once, no movie paired with itself (returns only the elements of the upper triangular region of the matrix)
        values.append(distance[pairs].mean()) # the mean distance of movie-pairs is saved for each user
    return float(np.mean(values)) # the mean over users is returned


def gini(recs, items) -> float:
    """Gini of how often each candidate movie is recommended. 0 = all equally often, 1 = one movie gets everything."""
    # How many lists each movie appears in; movies never recommended get 0
    counts = recs["item_id"].value_counts().reindex(items, fill_value=0) # counting how many times each movie appears in the recommendations, non-recommended movies are set to 0
    x = np.sort(counts.to_numpy())  # smallest count first
    n = len(x)
    i = np.arange(1, n + 1)  # rank of each count: 1, 2, ..., n
    return float(np.sum((2 * i - n - 1) * x) / (n * np.sum(x)))


def genre_distribution(table, genres) -> pd.DataFrame: # computes for each user the share of movies per genre
    """One row per user: the share of each genre among the user's movies.
    A movie with three genres gives each of them 1/3, so every movie counts equally.
    """
    n_genres = genres.sum(axis=1)  # calculates how many genres each movie has
    shares = genres.div(n_genres, axis=0)  # divides each movie's row by its own count, so the row sums to 1
    movie_shares = shares.loc[table["item_id"]]  # one row per movie in the table
    # Average the movies' shares within each user
    return movie_shares.groupby(table["user_id"].to_numpy()).mean()


def calibration_error(recs, fit, genres) -> float: # measures how well the genre distribution of the recommendation list matches the genre distribution of the user's viewing history
    """KL divergence between the user's genre distribution (from fit) and the list's, averaged over users."""
    q = genre_distribution(recs, genres)  # The genre distribution of each user's recommendation list: a table with one row per user and one column per genre.
    p = genre_distribution(fit, genres).loc[q.index]  # The same table, but with respect to the user's viewing history. Selects only the users that appear in table q and sets them to the same order
    q_smooth = (1 - alfa) * q + alfa * p  # Steck's smoothing: 99% of the top10 distribution and 1% of the viewing history distribution (to prevent division with 0)
    # KL(p || q) = sum over genres of p * log(p / q)
    ratio = (p / q_smooth).where(p > 0, 1)  # For each genre is calculated p/q: "how many times larger the genre's share is in history compared to the top-list". Genres with p = 0 get log(1) = 0, so they add nothing
    terms = p * np.log(ratio) # The KL terms p * log(p/q)
    return float(terms.sum(axis=1).mean()) # Sum over genres gives each user's KL divergence, and the mean over users gives one number