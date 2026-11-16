import numpy as np
import pandas as pd

from recsys.split import RELEVANT_MIN_RATING

def precision_at_k(predictions, ground_truth, k=10):
    """
    Calculates the Precision@k value for one user.
    Predictions is an ordered list of recommended items.
    Ground truth is a list or set of items that user liked.
    """
 
    # First k items from predictions.
    top_k_preds = predictions[:k]
    # Calculates how many recommendations were correct.
    hits = len(set(top_k_preds).intersection(set(ground_truth)))
    # Returns the success rate.
    return hits / k

def recall_at_k(predictions, ground_truth, k=10):
    """
    Calculates Recall@k value for one user.
    Predictions is an ordered list of recommended items.
    Ground truth is a list or set of items that user liked.
    """
    # Checks if user has zero movies in his/her item list.
    if len(ground_truth) == 0:
        return 0.0
    # First k items from predictions.
    top_k_preds = predictions[:k]
    # Calculates how many recommendations were correct.
    hits = len(set(top_k_preds).intersection(set(ground_truth)))
    # Returns the success rate.
    return hits / len(ground_truth)

def ndcg_at_k(predictions, ground_truth, k=10):
    """
    Calculates NDCG@k value for one user.
    Pay attention to the exact order of the matches.
    Predictions is an ordered list of recommended items.
    Ground truth is a list or set of items that user liked.
    """
    # First k items from predictions.
    top_k_preds = predictions[:k]
    # The variable dcg, in which calculating the points.
    dcg = 0.0
    # Go through the recommendation list in order.
    for i, pred in enumerate(top_k_preds):
        if pred in ground_truth:
            # Counting points for dcg with formula: 1 / log2(i+2).
            dcg += 1.0 / np.log2(i + 2)
            
    # Calculates the ideal dcg for perfect order.
    idcg = 0.0
    # The maximum number of hits one can expect from the list.
    ideal_hits = min(len(ground_truth), k)
    for i in range(ideal_hits):
        # Counting points for ideal dcg with formula: 1 / log2(i+2).
        idcg += 1.0 / np.log2(i + 2)
        
    # User has empty list.
    if idcg == 0.0:
        return 0.0
    # Returns the success rate.
    return dcg / idcg

def catalogue_coverage(all_predictions, full_catalogue, k=10):
    """
    Calculates the Catalogue Coverage metric.
    Measures the proportion of all movies that were recommended at least once.
    All predictions is a list containing the Top-K recommendations for all users.
    Full catalogue is a list of all unique movies in the system.
    """
    # Collects all unique movies that are recommended.
    recommended_items = set()
    # Go through the recommendations for all users.
    for preds in all_predictions:
        # Saves the top-k movies from each user.
        recommended_items.update(preds[:k])
        
    # Calculates the number of unique movies.
    num_unique_recommended = len(recommended_items.intersection(set(full_catalogue)))
    # Calculates the total number of movies.
    num_total_items = len(set(full_catalogue))
    
    # Zero movies.
    if num_total_items == 0:
        return 0.0
    # The number of recommendations for all films.
    return num_unique_recommended / num_total_items

def ranked_lists(recs: pd.DataFrame) -> dict:
    """Each user's recommended items as a list in rank order."""
    return (recs.sort_values(["user_id", "rank"])
                .groupby("user_id")["item_id"].apply(list).to_dict())

def per_user_metrics(recs: pd.DataFrame, target: pd.DataFrame, users: pd.Index,
                     k: int = 10, min_rating: int = RELEVANT_MIN_RATING) -> pd.DataFrame:
    """
    Precision@k, Recall@k and NDCG@k for every evaluated user, one row per user.
    Ground truth is the user's held-out ratings of min_rating or more, the same rule
    evaluated_users() uses. A user with no recommendations scores 0 instead of being
    left out, so a model cannot look better by skipping users.
    """
    # Relevant held-out ratings, grouped into one set of movies per user.
    relevant = target[target["rating"] >= min_rating]
    ground_truth = relevant.groupby("user_id")["item_id"].apply(set).to_dict()
    predictions = ranked_lists(recs)

    rows = []
    for user in users:
        user_preds = predictions.get(user, [])
        user_truth = ground_truth.get(user, set())
        rows.append({
            f"Precision@{k}": precision_at_k(user_preds, user_truth, k),
            f"Recall@{k}": recall_at_k(user_preds, user_truth, k),
            f"NDCG@{k}": ndcg_at_k(user_preds, user_truth, k),
        })
    return pd.DataFrame(rows, index=users)

def evaluate(recs: pd.DataFrame, target: pd.DataFrame, users: pd.Index,
             items: pd.Index, k: int = 10,
             min_rating: int = RELEVANT_MIN_RATING) -> dict[str, float]:
    """
    Mean Precision@k, Recall@k and NDCG@k over the evaluated users, plus catalogue coverage.
    Recs has the columns user_id, item_id and rank, as every model returns them.
    """
    results = {name: float(value)
               for name, value in per_user_metrics(recs, target, users, k, min_rating).mean().items()}
    results["Coverage"] = catalogue_coverage(list(ranked_lists(recs).values()), items, k)
    return results

def print_results(name: str, results: dict[str, float]) -> None:
    """Prints one model's results, one metric per line."""
    print(name)
    for metric, value in results.items():
        print(f"  {metric:<13} {value:.4f}")
