"""Figures of the reported RS-FEDRAD results, drawn from ``results/tables``.

    python -m rsfedrad.plots
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsfedrad import config

METRICS = ["Accuracy", "Precision", "Recall", "F1-score"]
COLORS = ["#1A9993", "#6699CC", "#66CC66", "#CC4C4C"]


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(df, key, title, name, ylim):
    x = np.arange(len(df))
    width = 0.8 / len(METRICS)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (metric, color) in enumerate(zip(METRICS, COLORS)):
        bars = ax.bar(x + (k - 1.5) * width, df[metric], width, label=metric, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1, rotation=90)
    ax.set_xticks(x, df[key])
    ax.set_ylim(*ylim)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, name)


def rounds_plot(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for metric, color, marker in zip(METRICS, COLORS, "osD^"):
        ax.plot(df.Round, df[metric], marker=marker, color=color, label=metric)
    ax.set_xticks(df.Round)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Score (%)")
    ax.set_title("RS-FEDRAD performance across federated rounds (FedAvg, 10 clients)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "federated_rounds.png")


def scalability_plot(df):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for metric, color, marker in zip(METRICS, COLORS, "osD^"):
        ax.plot(df.Clients, df[metric], marker=marker, color=color, label=metric)
    ax.set_xticks(df.Clients, [f"N = {n}" for n in df.Clients])
    ax.set_ylim(98.0, 100.0)
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Average score over 10 rounds (%)")
    ax.set_title("Scalability of RS-FEDRAD")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, "scalability.png")


def main():
    t = config.TABLES_DIR
    rounds_plot(pd.read_csv(t / "federated_rounds.csv"))
    grouped_bars(pd.read_csv(t / "aggregation_strategies.csv"), "Strategy",
                 "Aggregation strategies (average over 10 rounds)", "aggregation_strategies.png", (97.5, 100))
    grouped_bars(pd.read_csv(t / "centralized_vs_federated.csv"), "Mode",
                 "Centralised vs federated training", "centralized_vs_federated.png", (98.5, 100))
    grouped_bars(pd.read_csv(t / "adversarial_vs_federated.csv"), "Mode",
                 "Federated vs adversarially trained model", "adversarial_vs_federated.png", (98.5, 100))
    scalability_plot(pd.read_csv(t / "scalability.csv"))


if __name__ == "__main__":
    main()
