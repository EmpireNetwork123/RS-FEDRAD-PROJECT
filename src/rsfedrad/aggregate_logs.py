"""Rebuild the round-wise federated results from the per-client logs.

The published Table 2 (per-round accuracy, precision, recall and F1-score) is
the unweighted mean of the ten client logs in ``results/logs/federated``.

    python -m rsfedrad.aggregate_logs
"""
import argparse
from pathlib import Path

import pandas as pd

from rsfedrad import config

COLUMNS = ["Accuracy", "Precision", "Recall", "F1-score"]


def aggregate(log_dir):
    files = sorted(Path(log_dir).glob("client_*_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"no client logs in {log_dir}")
    frames = [pd.read_csv(f)[COLUMNS].reset_index(drop=True) for f in files]
    table = (sum(frames) / len(frames) * 100).round(2)
    table.insert(0, "Round", range(1, len(table) + 1))
    return table, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", default=str(config.LOGS_DIR / "federated"))
    args = parser.parse_args()
    table, n = aggregate(args.log_dir)
    print(f"Mean over {n} clients:\n")
    print(table.to_string(index=False))
    print("\nAverage over rounds:", table[COLUMNS].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
