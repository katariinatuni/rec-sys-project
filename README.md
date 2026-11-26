# rec-sys-project

Course project for DATA.ML.360 Recommender Systems (Tampere University): building, evaluating and critiquing recommenders on MovieLens 1M.

## 1. Get the data

The dataset is **not** included in this repository: the MovieLens licence does not allow redistributing it.

1. Download `ml-1m.zip` from GroupLens: <https://grouplens.org/datasets/movielens/1m/>
2. Check that it is the same file we used. The SHA-256 must be
   `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`
3. Unzip it **next to** this repository, so the folders look like this:

   ```
   any-folder/
   ├── data/
   │   └── ml-1m/          ratings.dat, movies.dat, users.dat, README
   └── rec-sys-project/    this repository
   ```

   Any other location works too: pass its path with `--data-dir`.

## 2. Install

Requires **Python 3.12**. All dependencies are pinned in `requirements.txt`.

With plain pip, if Python 3.12 is already installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run

From the repository root, one command reproduces every Part 1–3 number, table and figure in about 13 minutes:

```bash
python run_all.py --data-dir ../data/ml-1m --out-dir output_part3
```

It runs the steps below in order. Each step can also be run on its own, and every command takes `--out-dir` to choose where the outputs go (the default is `output/`).

There are three output folders:

| Folder | Contents |
|---|---|
| `output/` | the Part 1 results as submitted |
| `output_part2/` | the Part 2 results as submitted: the Part 1 baselines again, plus the new models, breakdowns, sparsity ablation and paired comparisons |
| `output_part3/` | the Part 3 results: everything in `output_part2/` again, plus the MMR diversity sweep and the who-pays breakdown |

Always pass `--out-dir`. Without it the commands write to `output/` and replace the Part 1 tables.

Random seeds: the models are deterministic. The random steps are the bootstraps in `compare.py`, `breakdown.py`, `model_comparisons.py`, `mmr_sweep.py` and `who_pays.py`, which use seed 0, and the subsampling in `sparsity_ablation.py`, which uses seeds 0, 1 and 2.

### Data characterization (Part 1, report section 1.1)

```bash
PYTHONPATH=src python -m recsys.explore --data-dir ../data/ml-1m
```

This reproduces every Part 1 data-characterization number and figure in about 15 seconds:

| Output | Contents |
|---|---|
| `output/data_stats.json` | all numbers: counts, density, distributions, long tail, activity spikes, sign-up and batch analysis |
| `output/figures/activity_over_time.png` | ratings per week (first-week vs returning users) and cumulative share |
| `output/figures/long_tail_rank_frequency.png` | ratings per film and per user, ranked, log-log |
| `output/figures/rating_distribution.png` | share of ratings at each star value |
| `output/figures/page_time_by_batch_size.png` | time on a rating page against films rated on it |
| `output/figures/stars_by_batch_size.png` | mean stars by number of films rated in the same second |

### Preprocessing and temporal split (Part 1, report sections 1.2 and 1.3)

```bash
PYTHONPATH=src python -m recsys.split_stats --data-dir ../data/ml-1m
```

This splits the ratings by time and reproduces every number and table in sections 1.2 and 1.3 in about 2 seconds:

| Output | Contents |
|---|---|
| `output/split_stats.json` | all numbers: split periods, relevance threshold, item filter, 80/10/10 comparison, post-2000 users |
| `output/report_tables.html` | Table 1 (thresholds) and Table 2 (split); open in a browser and copy into Word |
| `output/report_tables.md` | the same two tables as Markdown |

The split cuts at 00:00 UTC on 1 Jan 2001 and 1 Jul 2001:

| Set | Ratings | Used for |
|---|---|---|
| Train | before 1 Jan 2001 | fitting models for validation |
| Validation | 1 Jan – 30 Jun 2001 | choosing hyperparameters |
| Test | from 1 Jul 2001 | final evaluation, with models refitted on train and validation |

### Models and results (Part 2)

```bash
PYTHONPATH=src python -m recsys.compare --data-dir ../data/ml-1m --out-dir output_part2 --period val
PYTHONPATH=src python -m recsys.compare --data-dir ../data/ml-1m --out-dir output_part2 --period test
PYTHONPATH=src python -m recsys.results --out-dir output_part2
```

This evaluates every variant of every model on validation and on test (about 1.5 minutes per period), chooses one variant per model on validation, and writes the results in the columns of the results template's "Part 2 - Models" sheet.

The models and the variants compared:

| Model | Paradigm | Variants |
|---|---|---|
| Item-based kNN (Part 1 baseline) | neighbourhood CF | adjusted cosine or cosine; k = 20, 30, 50 |
| User-based kNN | neighbourhood CF | Pearson; significance weighting gamma = 1 (none) or 30; k = 20, 30, 50 |
| Markov chain | sequential | window = 1, 5, 20; popularity weight = 0.0 or 0.3 |
| Most popular (Part 1 baseline) | non-personalised | all-time or a 7 / 14 / 30 / 60-day window; all ratings or ratings ≥ 4 |

Selection rule, on validation only: among a model's variants that are not clearly worse than the best variant in the comparison (the 95% paired-bootstrap interval of the NDCG@10 difference includes zero), take the one with the highest catalogue coverage. If none qualifies, take the model's highest NDCG@10.

| Output | Contents |
|---|---|
| `output_part2/compare_val.csv`, `output_part2/compare_test.csv` | every variant: Precision@10, Recall@10, NDCG@10, coverage, and the NDCG difference to the best variant with a 95% paired-bootstrap interval |
| `output_part2/results.csv` | the same results in the "Part 2 - Models" columns, with the variant of each model chosen on validation marked |

Each model also runs on its own and prints its validation metrics, for example:

```bash
PYTHONPATH=src python -m recsys.knn_user --data-dir ../data/ml-1m --k 50 --gamma 30
PYTHONPATH=src python -m recsys.markov --data-dir ../data/ml-1m
```

### Markov chain tuning (Part 2)

```bash
PYTHONPATH=src python -m recsys.markov_tuning --data-dir ../data/ml-1m --out-dir output_part2
```

This runs the validation-only search behind the Markov chain's fixed settings, in about 2 minutes. It is not part of `run_all.py`. The search covers 84 settings: window 1 / 5 / 20, state built from the last 1 / 5 / 20 items or the whole history, decay 1.0 / 0.8, popularity weight 0.0 / 0.3 and a minimum gap of 0 or 1 day between the two ratings of a pair. The chosen setting is the default in `markov.py`, and `compare.py` varies the window and the popularity weight around it.

| Output | Contents |
|---|---|
| `output_part2/markov_tuning_val.csv` | every setting with its validation metrics and the difference to the best setting |
| `output_part2/markov_chosen.json` | the chosen setting |

### Breakdowns, sparsity ablation and paired comparisons (Part 2)

```bash
PYTHONPATH=src python -m recsys.breakdown --data-dir ../data/ml-1m --out-dir output_part2
PYTHONPATH=src python -m recsys.sparsity_ablation --data-dir ../data/ml-1m --out-dir output_part2
PYTHONPATH=src python -m recsys.model_comparisons --data-dir ../data/ml-1m --out-dir output_part2
```

These three steps take the variant of each model chosen on validation, so run `compare.py` on validation first. They evaluate on test (about 1, 5 and 1 minutes).

| Output | Contents |
|---|---|
| `output_part2/breakdown_test.csv` | each model overall, by user activity (at most / more than the median number of fit ratings) and by test time (held-out ratings before / from their median timestamp), with a 95% bootstrap interval for NDCG@10; the columns of the "Part 2 - Disaggregated" sheet |
| `output_part2/ablation_test.csv` | each model fitted on 100%, 50% and 20% of the fit ratings (random subsamples, mean over seeds 0, 1 and 2); the columns of the "Part 2 - Sparsity" sheet |
| `output_part2/comparisons_test.csv` | every pair of models: the NDCG@10 difference with a 95% paired-bootstrap interval over users |

### Diversity re-ranking (Part 3, Track A)

```bash
PYTHONPATH=src python -m recsys.mmr_sweep --data-dir ../data/ml-1m --out-dir output_part3
PYTHONPATH=src python -m recsys.who_pays --data-dir ../data/ml-1m --out-dir output_part3
```

The base recommender is the weighted hybrid chosen in Part 2 (alpha = 0.4). For every user it gives its 100 best movies, and MMR re-ranks them into a top-10, one movie at a time, picking the candidate with the highest

    lambda * relevance - (1 - lambda) * (highest similarity to the movies already picked)

Relevance is the hybrid's score, min-max normalised per user. Similarity is the cosine of the movies' 18-dimensional 0/1 genre vectors from `movies.dat`, so the item distance is 1 − cosine. lambda = 1 gives the hybrid's own top-10 unchanged.

`mmr_sweep.py` runs lambda = 0, 0.1, …, 1.0 on validation and on test (about 2 minutes). Selection rule, on validation only: among the lambda values whose NDCG@10 is not clearly worse than the best lambda's (the 95% paired-bootstrap interval of the difference includes zero), take the one with the highest intra-list diversity. On test, every lambda is compared with lambda = 1. `who_pays.py` then takes the chosen lambda (and lambda = 0.5 for contrast) on test and shows who gains and who loses (about 1 minute). It reads `mmr_chosen.json`, so run `mmr_sweep.py` first.

The beyond-accuracy metrics are in `new_metrics.py`:

| Metric | Definition |
|---|---|
| Intra-list diversity (ILD) | mean genre distance (1 − cosine) over all pairs of movies in a list, averaged over users |
| Catalogue coverage | share of the kept items that appear in at least one top-10 list |
| Gini | inequality of how often each kept item is recommended (0 = every item equally often, 1 = one item gets every slot) |
| Calibration error | Steck's KL(p ‖ q̃) between the genre distribution p of the user's fit ratings and q̃ = 0.99 q + 0.01 p, where q is the genre distribution of the list (a movie with n genres gives 1/n to each); lower is better |

| Output | Contents |
|---|---|
| `output_part3/mmr_sweep_val.csv` | every lambda on validation: Precision@10, Recall@10, NDCG@10, ILD, coverage, Gini, calibration error, and the NDCG difference to the best lambda with a 95% paired-bootstrap interval and whether it is clearly worse |
| `output_part3/mmr_chosen.json` | the lambda chosen on validation |
| `output_part3/mmr_sweep_test.csv` | every lambda on test, with the NDCG difference to lambda = 1 and its 95% paired-bootstrap interval |
| `output_part3/part3A_diversity_rows.csv` | the test sweep in the columns of the results template's "Part 3A - Diversity" sheet; lambda = 1 is marked as the unmodified base and the chosen lambda is marked |
| `output_part3/figures/mmr_frontier.png` | the accuracy–diversity frontier: NDCG@10 against ILD, calibration error and coverage, one point per lambda, validation and test |
| `output_part3/who_pays_users_test.csv` | all users, and users split by activity (at most / more than the median number of fit ratings) and by taste breadth (at most / more than the median genre entropy of p): NDCG@10, ILD and calibration error at lambda = 1 and at the chosen lambda, with 95% paired-bootstrap intervals for the differences |
| `output_part3/who_pays_items_test.csv` | at lambda = 1, the chosen lambda and 0.5: the share of recommendation slots that go to head items (the 10% most-rated kept items), the mean popularity percentile of the recommended movies, and the number of distinct movies recommended |
| `output_part3/who_pays_genres_test_lam*.csv` | each genre's share of the recommendation slots at lambda = 1 and at the chosen lambda, next to its share in the users' histories |
| `output_part3/who_pays_movers_test_lam*.csv` | the movies that lose and gain the most recommendation slots |

A single lambda also runs on its own and prints its metrics:

```bash
PYTHONPATH=src python -m recsys.mmr --data-dir ../data/ml-1m --lam 0.5 --period val
```

## Repository layout

```
run_all.py               runs every step above in order (except markov_tuning.py)
src/recsys/
├── data.py              loads ratings.dat, movies.dat and users.dat
├── explore.py           Part 1 data characterization (report section 1.1)
├── split.py             temporal split, evaluated users and item filter (sections 1.2-1.3)
├── split_stats.py       numbers and tables for sections 1.2-1.3
├── metrics.py           Precision@10, Recall@10, NDCG@10 and catalogue coverage
├── knn.py               item-based kNN baseline
├── popular.py           most-popular baseline
├── knn_user.py          user-based kNN (Pearson with significance weighting)
├── markov.py            Markov chain over each user's rating order
├── markov_tuning.py     validation search for the Markov chain's settings
├── hybrid.py            weighted hybrid of user-based kNN and the Markov chain
├── compare.py           evaluates every variant of every model on one period
├── results.py           chooses on validation; writes results.csv in the "Part 2 - Models" columns
├── breakdown.py         chosen models by user activity and by test time
├── sparsity_ablation.py chosen models on 100%, 50% and 20% of the fit ratings
├── model_comparisons.py paired comparison of every pair of chosen models
├── mmr.py               MMR re-ranking of the hybrid's top-100 with genre similarity (Part 3)
├── new_metrics.py       intra-list diversity, Gini and calibration error (Part 3)
├── mmr_sweep.py         lambda sweep, choice on validation, frontier figure (Part 3)
└── who_pays.py          who gains and who loses at the chosen lambda (Part 3)
output/                  Part 1 results as submitted
output_part2/            Part 2 results as submitted
output_part3/            Part 3 results, generated by the commands above
```
