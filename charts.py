"""Visualization: championship odds bar chart and advancement heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import config


def plot_title_odds(probs: pd.DataFrame, top_n: int = 15,
                    output_path: Path = config.RESULTS_DIR / "title_odds.png"):
    """Horizontal bar chart of championship probabilities for top N teams."""
    df = probs.nlargest(top_n, "champion").copy()
    df = df.sort_values("champion")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(df)))
    bars = ax.barh(df["team"], df["champion"] * 100, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df["champion"] * 100):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8.5)

    ax.set_xlabel("Championship probability (%)")
    ax.set_title(f"2026 FIFA World Cup — Title Odds (top {top_n})\n"
                 f"Dixon-Coles Monte Carlo, {config.N_SIMULATIONS:,} simulations",
                 fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved title odds chart -> {output_path}")


def plot_advancement_heatmap(probs: pd.DataFrame,
                              output_path: Path = config.RESULTS_DIR / "advancement_heatmap.png"):
    """Heatmap: teams × stages, colored by probability."""
    stages = ["group_advance", "r16", "qf", "sf", "final", "champion"]
    labels = ["Group\nAdvance", "R16", "QF", "SF", "Final", "Champion"]

    df = probs.set_index("team")[stages]
    # Sort by champion probability
    df = df.sort_values("champion", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.28)))
    im = ax.imshow(df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=7.5)
    ax.set_title("2026 FIFA World Cup — Advancement Probabilities\n"
                 f"Dixon-Coles, {config.N_SIMULATIONS:,} sims", fontsize=11)

    # Annotate cells
    for i in range(len(df)):
        for j in range(len(stages)):
            val = df.values[i, j]
            text_color = "black" if val < 0.6 else "white"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=6.5, color=text_color)

    plt.colorbar(im, ax=ax, label="Probability", shrink=0.6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved advancement heatmap -> {output_path}")
