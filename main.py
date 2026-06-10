"""
Entry point for the 2026 World Cup simulator.

Usage:
  python main.py                          # run full sim with cached data
  python main.py --refresh-elo            # re-scrape Elo ratings first
  python main.py --backtest               # run 2018/2022 backtests only
  python main.py --sims 50000             # override sim count
  python main.py --seed 123              # override random seed
  python main.py --no-host-bump          # disable host-nation advantage
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import config
from data_layer import load_results, compute_weights, get_elo_dict, all_teams, GROUPS
from dixon_coles import DixonColesModel, backtest
from engine import run_simulations
from charts import plot_title_odds, plot_advancement_heatmap


def parse_args():
    p = argparse.ArgumentParser(description="WC 2026 Monte Carlo simulator")
    p.add_argument("--refresh-elo", action="store_true", help="Re-scrape Elo ratings")
    p.add_argument("--backtest", action="store_true", help="Run backtests on WC 2018/2022")
    p.add_argument("--sims", type=int, default=config.N_SIMULATIONS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--no-host-bump", action="store_true", help="Disable host-nation bump")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_host_bump:
        config.HOST_BUMP = 0.0

    config.N_SIMULATIONS = args.sims
    config.RANDOM_SEED = args.seed

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading historical results...")
    if not config.HISTORICAL_RESULTS_CSV.exists():
        print(
            f"ERROR: {config.HISTORICAL_RESULTS_CSV} not found.\n"
            "Download from https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017\n"
            "and save as data/results.csv"
        )
        sys.exit(1)

    df = load_results()
    print(f"  Loaded {len(df):,} historical matches "
          f"({df['date'].min().year}–{df['date'].max().year})")

    print("Loading Elo ratings...")
    try:
        elo_dict = get_elo_dict(force_refresh=args.refresh_elo)
        print(f"  Loaded {len(elo_dict)} Elo ratings")
    except Exception as e:
        print(f"  WARNING: Could not fetch Elo ratings: {e}")
        print("  Continuing without Elo (pure Dixon-Coles)")
        elo_dict = {}

    # ── Backtest ─────────────────────────────────────────────────────────────
    if args.backtest:
        print("\nRunning backtests...")
        weights = compute_weights(df)
        model_ref = DixonColesModel(elo_dict=elo_dict)
        for year in [2018, 2022]:
            result = backtest(df, "FIFA World Cup", year, model_ref)
            print(f"  WC {year}: n={result.get('n_matches',0)}, "
                  f"log_loss={result.get('log_loss','N/A')}, "
                  f"brier={result.get('brier_score','N/A')}")
        return

    # Filter to matches where both teams are WC participants — faster fit,
    # less noise from obscure teams, better convergence.
    wc_teams = set(all_teams())
    df_fit = df[df["home"].isin(wc_teams) & df["away"].isin(wc_teams)].reset_index(drop=True)
    print(f"  Using {len(df_fit):,} WC-team matches (filtered from {len(df):,} total)")

    # ── Fit Dixon-Coles ──────────────────────────────────────────────────────
    print("\nFitting Dixon-Coles model...")
    t0 = time.time()
    weights = compute_weights(df_fit)
    model = DixonColesModel(elo_dict=elo_dict)
    model.fit(df_fit, weights)
    print(f"  Fit complete in {time.time()-t0:.1f}s")

    # Warn if any tournament teams are missing from the model
    missing = [t for t in all_teams() if t not in model.attack]
    if missing:
        print(f"  WARNING: {len(missing)} tournament teams not in training data: {missing}")
        print("  They will use zero attack/defense (league-average strength)")
        # Register them with zero params
        for t in missing:
            model.attack[t] = 0.0
            model.defense[t] = 0.0
        model._teams = list(set(model._teams) | set(missing))

    # ── Run simulations ──────────────────────────────────────────────────────
    print(f"\nRunning {args.sims:,} simulations...")
    t0 = time.time()
    probs = run_simulations(model, elo_dict, n=args.sims, seed=args.seed)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({args.sims/elapsed:.0f} sims/sec)")

    # ── Save outputs ─────────────────────────────────────────────────────────
    config.RESULTS_DIR.mkdir(exist_ok=True)

    out_csv = config.RESULTS_DIR / "team_probabilities.csv"
    probs.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Group expected standings
    group_rows = []
    for g, teams in GROUPS.items():
        for t in teams:
            row = probs[probs["team"] == t].iloc[0] if (probs["team"] == t).any() else {}
            group_rows.append({
                "group": g,
                "team": t,
                "p_advance": row.get("group_advance", 0.0) if isinstance(row, pd.Series) else 0.0,
            })
    group_df = pd.DataFrame(group_rows).sort_values(["group", "p_advance"], ascending=[True, False])
    group_csv = config.RESULTS_DIR / "group_tables.csv"
    group_df.to_csv(group_csv, index=False)
    print(f"Saved {group_csv}")

    # ── Print top 15 ─────────────────────────────────────────────────────────
    print("\n--- Championship Probabilities (top 15) ---")
    print(f"{'Team':<25} {'Champion':>9} {'Final':>7} {'SF':>7} {'QF':>7} {'R16':>7}")
    print("-" * 65)
    for _, row in probs.head(15).iterrows():
        print(f"{row['team']:<25} {row['champion']:>8.1%} {row['final']:>6.1%} "
              f"{row['sf']:>6.1%} {row['qf']:>6.1%} {row['r16']:>6.1%}")

    # ── Charts ───────────────────────────────────────────────────────────────
    if not args.no_charts:
        print("\nGenerating charts...")
        plot_title_odds(probs)
        plot_advancement_heatmap(probs)
        print("Charts saved to results/")


if __name__ == "__main__":
    main()
