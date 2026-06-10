"""
Daily underdog value-bet finder for the 2026 FIFA World Cup.

Usage:
  python daily_bets.py                  # today's matches
  python daily_bets.py --date 2026-06-15
  python daily_bets.py --days 3         # next 3 days
  python daily_bets.py --min-edge 10    # only show edge >= 10%
  python daily_bets.py --bookmaker bet365
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import config
from data_layer import load_results, compute_weights, get_elo_dict, all_teams
from dixon_coles import DixonColesModel

SPORT_KEY = "soccer_fifa_world_cup"
MARKETS   = "h2h"          # head-to-head (win/draw/win)
REGIONS   = "uk,eu,us"


# ── Odds API helpers ──────────────────────────────────────────────────────────

def fetch_odds(api_key: str) -> list[dict]:
    """Fetch all upcoming WC match odds from The Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  API requests remaining: {remaining}")
    return r.json()


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def best_odds(bookmakers: list[dict], outcome_name: str,
              preferred: str = "bet365") -> tuple[float, str]:
    """
    Return (best decimal odds, bookmaker name) for a given outcome.
    Prefers `preferred` bookmaker if available, otherwise takes the best line.
    """
    best = None
    best_book = ""
    preferred_odds = None
    preferred_found = False

    for bm in bookmakers:
        bm_key = bm.get("key", "")
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == outcome_name:
                    odds = outcome.get("price", 0)
                    if bm_key == preferred or bm.get("title", "").lower() == preferred.lower():
                        preferred_odds = odds
                        preferred_found = True
                    if best is None or odds > best:
                        best = odds
                        best_book = bm.get("title", bm_key)

    if preferred_found and preferred_odds:
        return preferred_odds, preferred.replace("williamhill", "William Hill").replace("paddypower", "Paddy Power").replace("betway", "Betway").replace("pinnacle", "Pinnacle")
    return (best or 2.0), best_book


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model() -> tuple[DixonColesModel, dict]:
    if not config.HISTORICAL_RESULTS_CSV.exists():
        print("ERROR: data/results.csv not found. Run main.py first.")
        sys.exit(1)

    df = load_results()
    elo_dict = {}
    try:
        elo_dict = get_elo_dict()
    except Exception:
        pass

    wc_teams = set(all_teams())
    df_fit = df[df["home"].isin(wc_teams) & df["away"].isin(wc_teams)].reset_index(drop=True)
    weights = compute_weights(df_fit)

    model = DixonColesModel(elo_dict=elo_dict)
    model.fit(df_fit, weights)

    missing = [t for t in all_teams() if t not in model.attack]
    for t in missing:
        model.attack[t] = 0.0
        model.defense[t] = 0.0
    if missing:
        model._teams = list(set(model._teams) | set(missing))

    return model, elo_dict


# ── Name normalisation ────────────────────────────────────────────────────────

# The Odds API uses slightly different team names — map to our canonical names
_API_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Draw": "Draw",
}

def normalise(name: str) -> str:
    return _API_NAME_MAP.get(name, name)


# ── Value bet analysis ────────────────────────────────────────────────────────

def analyse_match(event: dict, model: DixonColesModel,
                  min_edge: float, bookmaker: str) -> dict | None:
    """
    Compare model probabilities against bookmaker odds for one match.
    Returns a result dict if any value bets found, else None.
    """
    home_raw = event.get("home_team", "")
    away_raw = event.get("away_team", "")
    home = normalise(home_raw)
    away = normalise(away_raw)
    bookmakers = event.get("bookmakers", [])
    commence = event.get("commence_time", "")

    if not bookmakers:
        return None
    if home not in model.attack or away not in model.attack:
        return None

    # Model probabilities
    pred = model.predict(home, away, neutral=True,
                         host_bump_home=config.HOST_BUMP if home in config.HOST_NATIONS else 0.0,
                         host_bump_away=config.HOST_BUMP if away in config.HOST_NATIONS else 0.0)

    outcomes = [
        ("Home", home,  pred["p_home_win"]),
        ("Draw", "Draw", pred["p_draw"]),
        ("Away", away,  pred["p_away_win"]),
    ]

    bets = []
    for label, name, model_prob in outcomes:
        odds_name = home_raw if label == "Home" else (away_raw if label == "Away" else "Draw")
        dec_odds, book = best_odds(bookmakers, odds_name, bookmaker)
        imp = implied_prob(dec_odds)
        edge = (model_prob / imp - 1) * 100  # percentage edge

        bets.append({
            "label": label,
            "team": name,
            "model_prob": model_prob,
            "decimal_odds": dec_odds,
            "implied_prob": imp,
            "edge_pct": edge,
            "bookmaker": book,
        })

    # Flag value bets (positive edge on the underdog side)
    value_bets = [b for b in bets if b["edge_pct"] >= min_edge]

    return {
        "home": home,
        "away": away,
        "commence": commence,
        "bets": bets,
        "value_bets": value_bets,
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_match(match: dict, min_edge: float):
    home, away = match["home"], match["away"]
    dt = match["commence"]
    try:
        dt_parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        dt_str = dt_parsed.strftime("%b %d  %H:%M UTC")
    except Exception:
        dt_str = dt

    value_flag = "  *** VALUE ***" if match["value_bets"] else ""
    print(f"\n  {home} vs {away}  |  {dt_str}{value_flag}")
    print(f"  {'Outcome':<20} {'Model':>7} {'Odds':>7} {'Implied':>8} {'Edge':>8}  {'Book'}")
    print("  " + "-" * 62)

    for b in match["bets"]:
        edge_str = f"{b['edge_pct']:+.1f}%"
        marker = " <--" if b["edge_pct"] >= min_edge else ""
        print(f"  {b['team']:<20} {b['model_prob']:>6.1%} {b['decimal_odds']:>7.2f} "
              f"{b['implied_prob']:>7.1%} {edge_str:>8}  {b['bookmaker']}{marker}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Daily WC 2026 value bet finder")
    p.add_argument("--date", default=None, help="Date to check (YYYY-MM-DD), default: today")
    p.add_argument("--days", type=int, default=1, help="Number of days ahead to show")
    p.add_argument("--min-edge", type=float, default=5.0,
                   help="Minimum model edge %% to flag as value (default 5)")
    p.add_argument("--bookmaker", default="bet365",
                   help="Preferred bookmaker (default: bet365)")
    p.add_argument("--all", action="store_true",
                   help="Show all matches, not just value bets")
    return p.parse_args()


def main():
    args = parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    end_date = target_date + timedelta(days=args.days - 1)

    if not config.ODDS_API_KEY:
        print("ERROR: No Odds API key in config.py")
        sys.exit(1)

    print("Loading model...")
    model, elo_dict = load_model()

    print("Fetching odds from The Odds API...")
    events = fetch_odds(config.ODDS_API_KEY)
    print(f"  {len(events)} upcoming matches found")

    # Filter to target date range
    day_events = []
    for ev in events:
        try:
            ev_date = datetime.fromisoformat(
                ev["commence_time"].replace("Z", "+00:00")
            ).date()
        except Exception:
            continue
        if target_date <= ev_date <= end_date:
            day_events.append(ev)

    if not day_events:
        print(f"\nNo WC matches found for {target_date}"
              + (f" to {end_date}" if args.days > 1 else ""))
        print("(Odds may not be posted yet — try closer to the match date)")
        return

    # Analyse each match
    date_label = str(target_date) if args.days == 1 else f"{target_date} to {end_date}"
    print(f"\n{'='*65}")
    print(f"  WC 2026 Value Bets  |  {date_label}  |  min edge: {args.min_edge:.0f}%")
    print(f"{'='*65}")

    total_value = 0
    for ev in sorted(day_events, key=lambda e: e.get("commence_time", "")):
        result = analyse_match(ev, model, args.min_edge, args.bookmaker)
        if result is None:
            continue
        if args.all or result["value_bets"]:
            print_match(result, args.min_edge)
            total_value += len(result["value_bets"])

    if total_value == 0:
        print(f"\n  No value bets found at {args.min_edge:.0f}% edge threshold.")
        print("  Try --min-edge 3 for a lower bar, or --all to see all matches.")
    else:
        print(f"\n  {total_value} value bet(s) flagged above {args.min_edge:.0f}% edge.")

    print()


if __name__ == "__main__":
    main()
