"""
Bet tracker: logs today's value bets and resolves results from the scores API.
Run daily: python bet_tracker.py --log      # log today's value bets
           python bet_tracker.py --resolve  # fill in results for completed games
           python bet_tracker.py --summary  # print P&L summary
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import os
import config

# Allow GitHub Action to inject the API key via environment variable
if os.environ.get("ODDS_API_KEY"):
    config.ODDS_API_KEY = os.environ["ODDS_API_KEY"]

from data_layer import load_results, compute_weights, get_elo_dict, all_teams
from dixon_coles import DixonColesModel
from daily_bets import fetch_odds, analyse_match, best_odds, normalise

ADT = ZoneInfo("America/Halifax")
BETS_LOG = config.DATA_DIR / "bets_log.csv"
SPORT_KEY = "soccer_fifa_world_cup"

COLS = ["date", "match_date", "home", "away", "outcome",
        "odds", "edge_pct", "stake", "result", "pnl", "logged_at"]


def load_log() -> pd.DataFrame:
    if BETS_LOG.exists():
        df = pd.read_csv(BETS_LOG, dtype={"result": str, "pnl": str})
        for col in COLS:
            if col not in df.columns:
                df[col] = ""
        df["result"] = df["result"].fillna("").astype(str).str.strip()
        df["pnl"] = df["pnl"].fillna("").astype(str).str.strip()
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def save_log(df: pd.DataFrame):
    df.to_csv(BETS_LOG, index=False)


def load_model():
    df = load_results()
    elo_dict = get_elo_dict()
    wc_teams = set(all_teams())
    df_fit = df[df["home"].isin(wc_teams) & df["away"].isin(wc_teams)].reset_index(drop=True)
    model = DixonColesModel(elo_dict=elo_dict)
    model.fit(df_fit, compute_weights(df_fit))
    missing = [t for t in all_teams() if t not in model.attack]
    for t in missing:
        model.attack[t] = 0.0
        model.defense[t] = 0.0
    model._teams = list(set(model._teams) | set(missing))
    return model


def fetch_scores() -> dict[tuple[str, str], dict]:
    """Return {(home, away): {completed, home_score, away_score}} from scores API."""
    r = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/scores",
        params={"apiKey": config.ODDS_API_KEY, "daysFrom": 3},
        timeout=15,
    )
    r.raise_for_status()
    results = {}
    for ev in r.json():
        home = normalise(ev.get("home_team", ""))
        away = normalise(ev.get("away_team", ""))
        scores = {s["name"]: int(s["score"]) for s in (ev.get("scores") or [])}
        results[(home, away)] = {
            "completed": ev.get("completed", False),
            "home_score": scores.get(ev.get("home_team", ""), None),
            "away_score": scores.get(ev.get("away_team", ""), None),
        }
    return results


MAX_EDGE = 100.0  # cap — edges above this are model noise, not real value

def log_todays_bets(min_edge: float = 5.0, stake: float = 10.0):
    """Log all value bets for today and next 2 days."""
    model = load_model()
    events = fetch_odds(config.ODDS_API_KEY)
    log = load_log()
    today = date.today()
    new_rows = []

    for ev in events:
        ev_utc = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        match_date = ev_utc.astimezone(ADT).date()
        if match_date < today or match_date > today + timedelta(days=2):
            continue

        result = analyse_match(ev, model, min_edge, "williamhill")
        if not result:
            continue

        for b in [x for x in result["value_bets"] if x["edge_pct"] <= MAX_EDGE]:
            # Skip if already logged
            exists = (
                (log["home"] == result["home"]) &
                (log["away"] == result["away"]) &
                (log["outcome"] == b["team"])
            ).any()
            if exists:
                continue

            new_rows.append({
                "date": str(today),
                "match_date": str(match_date),
                "home": result["home"],
                "away": result["away"],
                "outcome": b["team"],
                "odds": round(b["decimal_odds"], 2),
                "edge_pct": round(b["edge_pct"], 1),
                "stake": stake,
                "result": None,
                "pnl": None,
                "logged_at": datetime.now().isoformat(),
            })

    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        save_log(log)
        print(f"Logged {len(new_rows)} new value bet(s).")
        for r in new_rows:
            print(f"  {r['home']} vs {r['away']} | {r['outcome']} @ {r['odds']} (edge {r['edge_pct']:+.1f}%)")
    else:
        print("No new value bets to log.")


def resolve_results():
    """Fill in results and P&L for completed games."""
    log = load_log()
    if log.empty:
        print("No bets logged yet.")
        return

    scores = fetch_scores()
    updated = 0

    for i, row in log.iterrows():
        if pd.notna(row.get("result")) and row["result"] != "":
            continue  # already resolved

        home, away = str(row["home"]), str(row["away"])
        score = scores.get((home, away)) or scores.get((away, home))
        if not score or not score["completed"]:
            continue

        hs, as_ = score["home_score"], score["away_score"]
        if hs is None or as_ is None:
            continue

        # Determine actual outcome
        if hs > as_:
            actual = home
        elif as_ > hs:
            actual = away
        else:
            actual = "Draw"

        outcome = str(row["outcome"])
        won = (outcome == actual)
        stake = float(row["stake"])
        odds = float(row["odds"])
        pnl = round(stake * (odds - 1), 2) if won else -stake

        log.at[i, "result"] = "W" if won else "L"
        log.at[i, "pnl"] = pnl
        updated += 1
        print(f"  {home} vs {away} ({hs}-{as_}) | {outcome} @ {odds} | {'WON' if won else 'LOST'} | P&L: {pnl:+.2f}")

    if updated:
        save_log(log)
        print(f"\nResolved {updated} bet(s).")
    else:
        print("No new results to resolve.")


def print_summary():
    """Print full P&L summary."""
    log = load_log()
    if log.empty:
        print("No bets logged.")
        return

    resolved = log[log["result"].notna() & (log["result"] != "")]
    pending = log[log["result"].isna() | (log["result"] == "")]

    print(f"\n{'='*55}")
    print(f"  WC 2026 Bet Tracker — P&L Summary")
    print(f"{'='*55}")
    print(f"  Total bets logged:  {len(log)}")
    print(f"  Resolved:           {len(resolved)}")
    print(f"  Pending:            {len(pending)}")

    if not resolved.empty:
        wins = (resolved["result"] == "W").sum()
        losses = (resolved["result"] == "L").sum()
        total_staked = resolved["stake"].astype(float).sum()
        total_pnl = resolved["pnl"].astype(float).sum()
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

        print(f"\n  Record:   {wins}W - {losses}L")
        print(f"  Staked:   ${total_staked:.2f}")
        print(f"  Net P&L:  ${total_pnl:+.2f}")
        print(f"  ROI:      {roi:+.1f}%")

        print(f"\n  {'Match':<30} {'Bet':<22} {'Odds':>5} {'Edge':>7} {'P&L':>8}")
        print("  " + "-" * 75)
        for _, r in resolved.iterrows():
            match = f"{r['home']} vs {r['away']}"[:29]
            print(f"  {match:<30} {str(r['outcome']):<22} {float(r['odds']):>5.2f} "
                  f"{float(r['edge_pct']):>+6.1f}% {float(r['pnl']):>+8.2f}")

    if not pending.empty:
        print(f"\n  Pending bets:")
        for _, r in pending.iterrows():
            print(f"    {r['home']} vs {r['away']} | {r['outcome']} @ {r['odds']} (match {r['match_date']})")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="store_true", help="Log today's value bets")
    p.add_argument("--resolve", action="store_true", help="Resolve completed game results")
    p.add_argument("--summary", action="store_true", help="Print P&L summary")
    p.add_argument("--all", action="store_true", help="Run all three steps")
    args = p.parse_args()

    if args.all or args.log:
        print("Logging value bets...")
        log_todays_bets()
    if args.all or args.resolve:
        print("Resolving results...")
        resolve_results()
    if args.all or args.summary:
        print_summary()
    if not any([args.log, args.resolve, args.summary, args.all]):
        p.print_help()
