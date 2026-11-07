"""Reproduce every Part 1 number, table and figure with one command.

    python run_all.py --data-dir ../data/ml-1m

Runs these steps in order (each one also runs on its own, see the README):
  1. recsys.explore      data characterization -> data_stats.json, figures/
  2. recsys.split_stats  temporal split and thresholds -> split_stats.json, report_tables.*
  3. recsys.compare      baselines on validation, then on test -> compare_val.csv, compare_test.csv
  4. recsys.results      choice on validation -> results.csv (results template columns)
All outputs go to --out-dir (default: output/ in this repository). Takes about a minute.
Everything is deterministic; the only random seed is the bootstrap's (0, in compare.py).
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


if __name__ == "__main__":
    main()
