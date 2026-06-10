"""Central configuration — edit here for sensitivity testing."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# ── Data sources ──────────────────────────────────────────────────────────────
HISTORICAL_RESULTS_CSV = DATA_DIR / "results.csv"       # Kaggle martj42 dataset
ELO_CACHE_CSV = DATA_DIR / "elo_ratings.csv"

# ── Model parameters ─────────────────────────────────────────────────────────
DECAY_HALF_LIFE_YEARS = 2.5          # exponential time-decay for Dixon-Coles fit
MIN_WEIGHT = 0.01                     # floor weight for very old matches
ELO_BLEND_WEIGHT = 0.3               # weight on Elo-implied mu vs. DC fitted mu (0 = pure DC)

# ── Host advantage ────────────────────────────────────────────────────────────
HOST_BUMP = 0.15                      # extra xG added to host-nation attack for group stage
HOST_NATIONS = {"United States", "Canada", "Mexico"}

# ── Simulation ────────────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000
RANDOM_SEED = 42                      # None to disable

# ── Penalty shootout ─────────────────────────────────────────────────────────
# True = pure 50/50; False = Elo-tilted (higher-rated team has slight edge)
SHOOTOUT_FIFTY_FIFTY = False
SHOOTOUT_ELO_SCALE = 400.0           # Elo points → ±5 % win probability shift

# ── Ratings date (for labelling cached scrape) ───────────────────────────────
ELO_RATINGS_DATE = "2026-06-09"

# ── The Odds API (optional calibration) ──────────────────────────────────────
# Read from Streamlit secrets when deployed, fall back to local value
try:
    import streamlit as st
    ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "35dd97f59712d7bea34a9750d27d7e95")
except Exception:
    ODDS_API_KEY = "35dd97f59712d7bea34a9750d27d7e95"
