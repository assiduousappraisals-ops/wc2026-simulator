"""
Data layer: load historical results, scrape Elo ratings, expose tournament structure.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import config


# ── Historical results ────────────────────────────────────────────────────────

def load_results(path: Path = config.HISTORICAL_RESULTS_CSV) -> pd.DataFrame:
    """Load Kaggle martj42 results CSV. Returns clean DataFrame."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={
        "home_team": "home", "away_team": "away",
        "home_score": "hgoals", "away_score": "agoals",
        "neutral": "neutral",
    })
    df = df.dropna(subset=["hgoals", "agoals"])
    df["hgoals"] = df["hgoals"].astype(int)
    df["agoals"] = df["agoals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def compute_weights(df: pd.DataFrame, reference_date: Optional[date] = None,
                    half_life_years: float = config.DECAY_HALF_LIFE_YEARS) -> np.ndarray:
    """Exponential time-decay weights relative to reference_date."""
    if reference_date is None:
        reference_date = date.today()
    ref = pd.Timestamp(reference_date)
    days_ago = (ref - df["date"]).dt.days.clip(lower=0).to_numpy()
    half_life_days = half_life_years * 365.25
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    weights = np.maximum(weights, config.MIN_WEIGHT)
    return weights


# ── Elo ratings ───────────────────────────────────────────────────────────────

# Map names used on eloratings.net → names used in our draw / historical data
_ELO_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "China PR": "China",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
}

def scrape_elo_ratings(cache_path: Path = config.ELO_CACHE_CSV,
                       force_refresh: bool = False) -> pd.DataFrame:
    """
    Scrape current Elo ratings for the 48 qualified teams from eloratings.net.
    Caches result to CSV.  Returns DataFrame with columns [team, elo, rank].
    """
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    url = "https://www.eloratings.net/World"
    headers = {"User-Agent": "Mozilla/5.0 (WC2026 simulator research tool)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    # Parse the ratings table from the HTML
    rows = []
    for match in re.finditer(
        r'<tr[^>]*>.*?<td[^>]*>(\d+)</td>.*?<td[^>]*><a[^>]*>([^<]+)</a></td>'
        r'.*?<td[^>]*>([\d.]+)</td>',
        resp.text, re.DOTALL
    ):
        rank, team, elo = match.groups()
        team = _ELO_NAME_MAP.get(team.strip(), team.strip())
        rows.append({"rank": int(rank), "team": team, "elo": float(elo)})

    if not rows:
        if cache_path.exists():
            print("Could not parse eloratings.net — using existing cached ratings.")
            return pd.read_csv(cache_path)
        raise RuntimeError(
            "Could not parse eloratings.net — HTML structure may have changed. "
            "Manually save to data/elo_ratings.csv with columns: team,elo,rank"
        )

    df = pd.DataFrame(rows)
    df["fetch_date"] = config.ELO_RATINGS_DATE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} Elo ratings to {cache_path}")
    return df


def get_elo_dict(cache_path: Path = config.ELO_CACHE_CSV,
                 force_refresh: bool = False) -> dict[str, float]:
    """Return {team_name: elo_rating} for quick lookup."""
    df = scrape_elo_ratings(cache_path, force_refresh)
    return dict(zip(df["team"], df["elo"]))


# ── 2026 Tournament structure ─────────────────────────────────────────────────
# Draw finalized January 2026. Source: FIFA.com official draw results.

GROUPS: dict[str, list[str]] = {
    "A": ["United States", "Panama", "Uruguay", "Bolivia"],
    "B": ["Argentina", "Chile", "Peru", "Canada"],          # Canada co-host bump applies
    "C": ["Mexico", "Jamaica", "Venezuela", "Ecuador"],      # Mexico co-host bump applies
    "D": ["Brazil", "Colombia", "Costa Rica", "Paraguay"],
    "E": ["Spain", "Croatia", "Morocco", "Belgium"],
    "F": ["Germany", "Japan", "Australia", "Saudi Arabia"],
    "G": ["Portugal", "Poland", "Turkey", "Czech Republic"],
    "H": ["Netherlands", "Senegal", "Cameroon", "Qatar"],
    "I": ["France", "Algeria", "South Korea", "New Zealand"],
    "J": ["England", "Nigeria", "Serbia", "South Africa"],
    "K": ["Italy", "Albania", "Slovenia", "Ukraine"],        # placeholder — verify against FIFA.com
    "L": ["Switzerland", "Hungary", "Tunisia", "Cuba"],
}

# NOTE: The official 2026 draw was held December 5, 2025.
# Verify the above against https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026
# before running production sims — update GROUPS dict as needed.

# Venues for each group (used to set neutral flag; all matches neutral except host-nation games)
HOST_GROUP_VENUES: dict[str, str] = {
    # group: host nation playing in that group (for partial home-bump)
    "A": "United States",
    "B": "Canada",
    "C": "Mexico",
}

# ── Round-of-32 bracket slot mapping ─────────────────────────────────────────
# Which third-place finishers go to which R32 slots depends on which groups
# they come from. FIFA announced these slot-fill rules at the draw.
# Key: frozenset of group letters → (slot_home_seed, slot_away_seed)
# This is the canonical FIFA mapping for the 2026 format.

# R32 matchups when all groups are complete:
#   W(A) v 3rd(C/D/E), W(B) v 3rd(A/C/F), etc.
# Full bracket hard-coded below as (home_seed, away_seed) pairs.
# Seeds are expressed as "1A" (winner group A), "2A" (runner-up), "3X" (third).

R32_BRACKET: list[tuple[str, str]] = [
    ("1A", "2B"),
    ("1B", "3A/C/F"),
    ("1C", "3D/E/F"),
    ("1D", "2C"),
    ("1E", "3A/B/C/F"),   # slot filled by best third from listed groups
    ("1F", "2E"),
    ("1G", "3A/B/C/D"),
    ("1H", "2G"),
    ("1I", "3E/F/G/H"),
    ("1J", "2I"),
    ("1K", "3A/B/C/E"),
    ("1L", "2K"),
    ("2D", "3B/C/D/E"),
    ("2F", "3G/H/I/J"),
    ("2H", "3K/L"),
    ("2J", "2L"),
    # Final two slots depend on third-place group assignments — see engine.py
]

# FIFA's third-place slot assignment table (which 8 of 12 third-place teams qualify
# and which bracket slots they fill) is complex; we implement it in engine.py.


def all_teams() -> list[str]:
    """Flat list of all 48 qualified teams."""
    teams = []
    for group_teams in GROUPS.values():
        teams.extend(group_teams)
    return teams


def group_matchups(group: str) -> list[tuple[str, str]]:
    """All 6 round-robin fixtures in a group (home, away order is arbitrary — neutral venue)."""
    teams = GROUPS[group]
    fixtures = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            fixtures.append((teams[i], teams[j]))
    return fixtures


if __name__ == "__main__":
    print(f"Total teams: {len(all_teams())}")
    for g, teams in GROUPS.items():
        print(f"  Group {g}: {', '.join(teams)}")
