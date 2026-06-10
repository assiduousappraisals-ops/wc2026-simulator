"""
Smoke test: generate a tiny synthetic dataset, fit the model, run 200 sims.
Run with: python smoke_test.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

# Patch config before importing anything else
import config
config.N_SIMULATIONS = 200
config.RANDOM_SEED = 1
config.ELO_BLEND_WEIGHT = 0.0   # pure DC for smoke test

from data_layer import compute_weights, all_teams, GROUPS
from dixon_coles import DixonColesModel
from engine import run_simulations

# ── Build synthetic match history covering all tournament teams ───────────────
rng = np.random.default_rng(42)
teams = all_teams()
rows = []
dates = pd.date_range("2020-01-01", "2026-01-01", periods=2000)
for date in dates:
    h, a = rng.choice(teams, 2, replace=False)
    hg = int(rng.poisson(1.3))
    ag = int(rng.poisson(1.1))
    rows.append({"date": date, "home": h, "away": a,
                 "hgoals": hg, "agoals": ag,
                 "neutral": True, "tournament": "Friendly"})
df = pd.DataFrame(rows)

print(f"Synthetic dataset: {len(df)} matches, {len(teams)} teams")

weights = compute_weights(df)
model = DixonColesModel()
model.fit(df, weights)

probs = run_simulations(model, elo_dict={}, n=200, seed=1)
print(f"\nSimulations complete. Top 5 title contenders:")
print(probs[["team", "champion", "final", "sf"]].head(5).to_string(index=False))
print("\nAll probabilities sum to 1 (champion):", round(probs["champion"].sum(), 3))
print("\nSmoke test PASSED")
