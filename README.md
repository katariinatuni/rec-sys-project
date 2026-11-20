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

From the repository root, one command reproduces every Part 2 number, table and figure in about 10 minutes:

```bash
python run_all.py --data-dir ../data/ml-1m --out-dir output_part2
```

It runs the steps below in order. Each step can also be run on its own, and every command takes `--out-dir` to choose where the outputs go (the default is `output/`).

There are two output folders:

| Folder | Contents |
|---|---|
| `output/` | the Part 1 results as submitted (the code on branch `main` reproduces them) |
| `output_part2/` | the Part 2 results: the Part 1 baselines again, plus the new models, breakdowns, sparsity ablation and paired comparisons |

Keep `--out-dir output_part2` in the Part 2 commands. Without it they write to `output/` and replace the Part 1 tables.

Random seeds: the models are deterministic. The random steps are the bootstraps in `compare.py`, `breakdown.py` and `model_comparisons.py`, which use seed 0, and the subsampling in `sparsity_ablation.py`, which uses seeds 0, 1 and 2.

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
├── compare.py           evaluates every variant of every model on one period
├── results.py           chooses on validation; writes results.csv in the "Part 2 - Models" columns
├── breakdown.py         chosen models by user activity and by test time
├── sparsity_ablation.py chosen models on 100%, 50% and 20% of the fit ratings
└── model_comparisons.py paired comparison of every pair of chosen models
output/                  Part 1 results as submitted
output_part2/            Part 2 results, generated by the commands above
```
