"""Tests for group tiebreaker logic."""
import pytest
import numpy as np
from engine import TeamStats, sort_group, MatchResult


def make_rng():
    return np.random.default_rng(0)


def _stats(team, pts, gd, gf, group="A"):
    s = TeamStats(team, group)
    # Reverse-engineer wins/draws/losses from points and gd
    s.gf = gf
    s.ga = gf - gd
    wins = pts // 3
    draws = pts % 3
    losses = 3 - wins - draws
    s.wins, s.draws, s.losses = wins, draws, losses
    s.played = 3
    return s


def test_sort_by_points():
    stats = {
        "Brazil": _stats("Brazil", 9, 5, 7),
        "Argentina": _stats("Argentina", 6, 2, 4),
        "Chile": _stats("Chile", 3, -2, 2),
        "Peru": _stats("Peru", 0, -5, 0),
    }
    result = sort_group(stats, [], make_rng())
    assert result == ["Brazil", "Argentina", "Chile", "Peru"]


def test_sort_by_gd_when_points_tied():
    stats = {
        "Spain": _stats("Spain", 6, 4, 6),
        "Germany": _stats("Germany", 6, 2, 5),
        "France": _stats("France", 3, -2, 2),
        "Italy": _stats("Italy", 3, -4, 1),
    }
    result = sort_group(stats, [], make_rng())
    assert result[0] == "Spain"
    assert result[1] == "Germany"


def test_sort_by_gf_when_points_gd_tied():
    stats = {
        "A": _stats("A", 6, 2, 5),
        "B": _stats("B", 6, 2, 3),  # same pts/gd, fewer GF
        "C": _stats("C", 3, 0, 2),
        "D": _stats("D", 3, -4, 0),
    }
    result = sort_group(stats, [], make_rng())
    assert result[0] == "A"
    assert result[1] == "B"


def test_h2h_tiebreaker():
    """A and B both 6 pts, same GD, same GF — A beat B head-to-head."""
    stats = {
        "A": _stats("A", 6, 2, 4),
        "B": _stats("B", 6, 2, 4),
        "C": _stats("C", 3, 0, 2),
        "D": _stats("D", 3, -4, 0),
    }
    # A beat B 1-0
    results = [MatchResult("A", "B", 1, 0)]
    ordered = sort_group(stats, results, make_rng())
    assert ordered[0] == "A"
    assert ordered[1] == "B"


def test_all_tied_no_infinite_loop():
    """Completely tied group — should return a permutation without hanging."""
    stats = {
        "A": _stats("A", 3, 0, 3),
        "B": _stats("B", 3, 0, 3),
        "C": _stats("C", 3, 0, 3),
        "D": _stats("D", 3, 0, 3),
    }
    result = sort_group(stats, [], make_rng())
    assert set(result) == {"A", "B", "C", "D"}
    assert len(result) == 4
