"""Tests for tournament engine: third-place ranking and bracket construction."""
import numpy as np
import pytest
from engine import (
    TeamStats, rank_third_place, _build_r32_bracket,
    simulate_penalties, MatchResult, knockout_winner
)
import config


def make_rng(seed=0):
    return np.random.default_rng(seed)


def _third(team, pts, gd, gf):
    s = TeamStats(team, "X")
    s.gf = gf
    s.ga = gf - gd
    wins = pts // 3
    draws = pts % 3
    losses = 3 - wins - draws
    s.wins, s.draws, s.losses = max(wins, 0), max(draws, 0), max(losses, 0)
    s.played = 3
    return s


def test_rank_third_place_returns_8():
    thirds = [(f"T{i}", _third(f"T{i}", i % 7, i % 3, i % 4)) for i in range(12)]
    best = rank_third_place(thirds, make_rng())
    assert len(best) == 8


def test_rank_third_place_best_points_first():
    thirds = [
        ("Best", _third("Best", 7, 5, 8)),
        ("Middle", _third("Middle", 4, 1, 3)),
    ] + [(f"T{i}", _third(f"T{i}", 2, -1, 1)) for i in range(10)]
    best = rank_third_place(thirds, make_rng())
    assert best[0] == "Best"


def test_r32_bracket_has_16_pairs():
    groups = {g: [f"{g}1", f"{g}2", f"{g}3", f"{g}4"]
              for g in "ABCDEFGHIJKL"}
    standings = {g: teams for g, teams in groups.items()}
    thirds = [f"T{i}" for i in range(8)]
    pairs = _build_r32_bracket(standings, thirds)
    assert len(pairs) == 16
    all_teams_in_bracket = [t for pair in pairs for t in pair]
    assert len(all_teams_in_bracket) == 32


def test_r32_bracket_all_teams_distinct():
    groups = {g: [f"{g}1", f"{g}2", f"{g}3", f"{g}4"]
              for g in "ABCDEFGHIJKL"}
    standings = {g: teams for g, teams in groups.items()}
    thirds = [f"T{i}" for i in range(8)]
    pairs = _build_r32_bracket(standings, thirds)
    all_teams = [t for pair in pairs for t in pair]
    # All 32 should be unique
    assert len(all_teams) == len(set(all_teams))


def test_penalties_returns_one_of_two_teams():
    elo = {"A": 1800, "B": 1600}
    results = [simulate_penalties("A", "B", elo, make_rng(s)) for s in range(100)]
    assert set(results) <= {"A", "B"}


def test_penalties_elo_favourite_wins_more():
    """Higher-Elo team should win more than 50% of shootouts."""
    elo = {"A": 2000, "B": 1400}
    config.SHOOTOUT_FIFTY_FIFTY = False
    wins = sum(
        simulate_penalties("A", "B", elo, make_rng(s)) == "A"
        for s in range(1000)
    )
    assert wins > 500  # A should win more


def test_match_result_winner():
    r = MatchResult("Home", "Away", 2, 1)
    assert r.winner == "Home"
    r2 = MatchResult("Home", "Away", 1, 1)
    assert r2.winner is None
    r3 = MatchResult("Home", "Away", 0, 3)
    assert r3.winner == "Away"
