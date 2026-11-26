"""Reproduce every Part 1-3 number, table and figure with one command.

    python run_all.py --data-dir ../data/ml-1m --out-dir output_part3

Runs these steps in order (each one also runs on its own, see the README):
  Part 1
  1. recsys.explore           data characterization -> data_stats.json, figures/
  2. recsys.split_stats       temporal split and thresholds -> split_stats.json, report_tables.*
  Part 2
  3. recsys.compare           every model variant on validation, then on test -> compare_*.csv
  4. recsys.results           choice on validation -> results.csv ("Part 2 - Models" columns)
  5. recsys.breakdown         chosen models by user activity and test time -> breakdown_test.csv
  6. recsys.sparsity_ablation chosen models on 100 / 50 / 20 % of the fit ratings -> ablation_test.csv
  7. recsys.model_comparisons every pair of chosen models -> comparisons_test.csv
  Part 3 (Track A: diversity)
  8. recsys.mmr_sweep         MMR lambda sweep, choice on validation, test -> mmr_sweep_*.csv,
                              mmr_chosen.json, part3A_diversity_rows.csv, figures/mmr_frontier.png
  9. recsys.who_pays          who pays for the chosen lambda, on test -> who_pays_*.csv
All outputs go to --out-dir (default: output/ in this repository). Takes about 13 minutes.
The models are deterministic. The random steps are the bootstraps (seed 0) and the subsampling
in sparsity_ablation.py (seeds 0, 1 and 2).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # the repository root, wherever the command is run from


def run(module: str, *args: str) -> None:
    """Run one step as `python -m recsys.<module>` with src/ on the import path; stop on failure."""
    print(f"\n=== recsys.{module} {' '.join(args)}", flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", f"recsys.{module}", *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="folder with ratings.dat, movies.dat, users.dat")
    parser.add_argument("--out-dir", default=str(ROOT / "output"), help="folder for all outputs")
    args = parser.parse_args()
    # Absolute paths, because the steps run with the repository root as working directory.
    data = ["--data-dir", str(Path(args.data_dir).resolve())]
    out = ["--out-dir", str(Path(args.out_dir).resolve())]

    run("explore", *data, *out)
    run("split_stats", *data, *out)
    for period in ["val", "test"]:  # validation first: the choice is made there
        run("compare", *data, *out, "--period", period)
    run("results", *out)
    run("breakdown", *data, *out)
    run("sparsity_ablation", *data, *out)
    run("model_comparisons", *data, *out)
    run("mmr_sweep", *data, *out)  # chooses lambda on validation, writes mmr_chosen.json
    run("who_pays", *data, *out)   # reads mmr_chosen.json, so it must run after mmr_sweep


if __name__ == "__main__":
    main()
