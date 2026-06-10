"""
Tournament engine: group stage tiebreakers, third-place ranking,
knockout bracket, and full Monte Carlo simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
from data_layer import GROUPS, HOST_GROUP_VENUES, all_teams, group_matchups


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    home: str
    away: str
    hgoals: int
    agoals: int

    @property
    def winner(self) -> Optional[str]:
        if self.hgoals > self.agoals:
            return self.home
        if self.agoals > self.hgoals:
            return self.away
        return None  # draw

    @property
    def loser(self) -> Optional[str]:
        w = self.winner
        if w is None:
            return None
        return self.away if w == self.home else self.home


@dataclass
class TeamStats:
    team: str
    group: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def gd(self) -> int:
        return self.gf - self.ga


# ── Group stage tiebreakers ───────────────────────────────────────────────────

def _h2h_record(teams: list[str], results: list[MatchResult]) -> dict[str, TeamStats]:
    """Build head-to-head mini table for a subset of teams."""
    h2h: dict[str, TeamStats] = {t: TeamStats(t, "") for t in teams}
    for r in results:
        if r.home in h2h and r.away in h2h:
            h2h[r.home].played += 1
            h2h[r.away].played += 1
            h2h[r.home].gf += r.hgoals
            h2h[r.home].ga += r.agoals
            h2h[r.away].gf += r.agoals
            h2h[r.away].ga += r.hgoals
            if r.hgoals > r.agoals:
                h2h[r.home].wins += 1
                h2h[r.away].losses += 1
            elif r.agoals > r.hgoals:
                h2h[r.away].wins += 1
                h2h[r.home].losses += 1
            else:
                h2h[r.home].draws += 1
                h2h[r.away].draws += 1
    return h2h


def sort_group(stats: dict[str, TeamStats],
               results: list[MatchResult],
               rng: np.random.Generator) -> list[str]:
    """
    Sort teams in a group by FIFA tiebreaker rules:
    1. Points  2. GD  3. GF  4. H2H points  5. H2H GD  6. H2H GF  7. Random draw
    """
    teams = list(stats.keys())

    def sort_key(team: str):
        s = stats[team]
        return (s.points, s.gd, s.gf)

    teams.sort(key=sort_key, reverse=True)

    # Re-sort tied clusters using head-to-head
    i = 0
    while i < len(teams):
        j = i + 1
        while j < len(teams) and sort_key(teams[j]) == sort_key(teams[i]):
            j += 1
        if j - i > 1:  # tied group
            tied = teams[i:j]
            h2h = _h2h_record(tied, results)
            tied.sort(
                key=lambda t: (
                    h2h[t].points, h2h[t].gd, h2h[t].gf,
                    rng.random()  # random draw as final tiebreaker
                ),
                reverse=True,
            )
            teams[i:j] = tied
        i = j

    return teams


# ── Third-place ranking and slot assignment ───────────────────────────────────

# Third-place teams are ranked across all 12 groups; best 8 advance.
# Tiebreaker: points → GD → GF → random
def rank_third_place(third_stats: list[tuple[str, TeamStats]],
                     rng: np.random.Generator) -> list[str]:
    """Return the 8 best third-placed teams, ranked."""
    third_stats.sort(
        key=lambda x: (x[1].points, x[1].gd, x[1].gf, rng.random()),
        reverse=True,
    )
    return [t for t, _ in third_stats[:8]]


# FIFA 2026 third-place slot assignment:
# The 8 best thirds come from specific groups; which bracket slot they fill
# depends on which groups they come from. This mapping was released by FIFA.
# Key: frozenset of groups of the 8 best thirds → slot assignments list
# For simplicity we use the positional assignment FIFA defined.
# The 16 R32 bracket slots (indexed 0..15) — slots expecting a third-place team:
# Slots 1, 2, 4, 6, 8, 12, 13, 14 receive a third-place team.
# The exact group→slot mapping:
_THIRD_SLOT_RULES: list[tuple[frozenset, list[int]]] = [
    # These are the FIFA-defined rules; they map the letter combination
    # of qualifying thirds to specific bracket slots.
    # Format: (frozenset of group letters, [slot_indices in R32_BRACKET order])
    # Simplified: assign in points order to the available third-place slots
]
# Since the exact FIFA 2026 table isn't yet public, we assign the 8 best thirds
# to the 8 third-place slots in bracket order (a common approximation).
THIRD_PLACE_R32_SLOTS = [1, 2, 4, 6, 8, 10, 12, 14]  # indices into R32 fixture list


# ── Match simulation utilities ────────────────────────────────────────────────

def _host_bump(team: str) -> float:
    return config.HOST_BUMP if team in config.HOST_NATIONS else 0.0


def simulate_match(model, home: str, away: str,
                   neutral: bool = True,
                   rng: np.random.Generator = None) -> MatchResult:
    """Simulate a single match using the Dixon-Coles model."""
    if rng is None:
        rng = np.random.default_rng()
    hbump = _host_bump(home)
    abump = _host_bump(away)
    hg, ag = model.sample_score(home, away, neutral=neutral,
                                 host_bump_home=hbump,
                                 host_bump_away=abump,
                                 rng=rng)
    return MatchResult(home, away, hg, ag)


def simulate_extra_time(model, home: str, away: str,
                         rng: np.random.Generator) -> MatchResult:
    """
    Extra time: sample ~0.3 xG per team (30 min period × reduced scoring rate).
    If still level after ET, go to penalties.
    """
    mu_h, mu_a = model.expected_goals(home, away, neutral=True,
                                       host_bump_home=_host_bump(home),
                                       host_bump_away=_host_bump(away))
    # ET is ~0.3 of a full match
    et_scale = 0.3
    hg = int(rng.poisson(mu_h * et_scale))
    ag = int(rng.poisson(mu_a * et_scale))
    return MatchResult(home, away, hg, ag)


def simulate_penalties(home: str, away: str,
                        elo_dict: dict[str, float],
                        rng: np.random.Generator) -> str:
    """Return winner of penalty shootout."""
    if config.SHOOTOUT_FIFTY_FIFTY:
        return rng.choice([home, away])
    elo_h = elo_dict.get(home, 1500.0)
    elo_a = elo_dict.get(away, 1500.0)
    diff = elo_h - elo_a
    p_home = 0.5 + diff / (2 * config.SHOOTOUT_ELO_SCALE)
    p_home = float(np.clip(p_home, 0.3, 0.7))
    return home if rng.random() < p_home else away


def knockout_winner(model, home: str, away: str,
                    elo_dict: dict[str, float],
                    rng: np.random.Generator) -> str:
    """Simulate a knockout match including ET and pens if needed."""
    r = simulate_match(model, home, away, neutral=True, rng=rng)
    if r.winner:
        return r.winner
    # Extra time
    et = simulate_extra_time(model, home, away, rng)
    if et.hgoals > et.agoals:
        return home
    if et.agoals > et.hgoals:
        return away
    # Penalties
    return simulate_penalties(home, away, elo_dict, rng)


# ── Single tournament simulation ─────────────────────────────────────────────

def simulate_tournament(model, elo_dict: dict[str, float],
                         rng: np.random.Generator,
                         locked_results: Optional[dict] = None) -> dict[str, list[str]]:
    """
    Simulate a full tournament. Returns dict mapping stage → list of teams
    that reached that stage (for aggregation across sims).

    locked_results: optional dict of already-played match results to lock in
                    (used for live-update mode). Not yet implemented — placeholder.
    """
    stage_reached: dict[str, set[str]] = {
        "group": set(),
        "r32": set(),
        "r16": set(),
        "qf": set(),
        "sf": set(),
        "final": set(),
        "champion": set(),
    }

    # ── Group stage ─────────────────────────────────────────────────────────
    group_results: dict[str, list[MatchResult]] = {}
    group_standings: dict[str, list[str]] = {}  # group → [1st, 2nd, 3rd, 4th]
    all_third: list[tuple[str, TeamStats]] = []

    for group_letter, group_teams in GROUPS.items():
        stats = {t: TeamStats(t, group_letter) for t in group_teams}
        results: list[MatchResult] = []
        host_nation = HOST_GROUP_VENUES.get(group_letter)

        for home, away in group_matchups(group_letter):
            neutral = True  # World Cup — all neutral
            r = simulate_match(model, home, away, neutral=neutral, rng=rng)
            results.append(r)

            # Update stats
            for team, gf, ga in [(home, r.hgoals, r.agoals), (away, r.agoals, r.hgoals)]:
                stats[team].played += 1
                stats[team].gf += gf
                stats[team].ga += ga
                if gf > ga:
                    stats[team].wins += 1
                elif gf == ga:
                    stats[team].draws += 1
                else:
                    stats[team].losses += 1

        sorted_teams = sort_group(stats, results, rng)
        group_standings[group_letter] = sorted_teams
        group_results[group_letter] = results
        all_third.append((sorted_teams[2], stats[sorted_teams[2]]))

        for t in sorted_teams:
            stage_reached["group"].add(t)

    # ── Third-place ranking ──────────────────────────────────────────────────
    best_thirds = rank_third_place(all_third, rng)

    # ── Build R32 field ──────────────────────────────────────────────────────
    # 24 group toppers/runners-up + 8 best thirds = 32 teams
    # We pair them using a simplified bracket: winners vs runners-up across groups
    # and inject thirds into the designated slots.
    r32_pairs = _build_r32_bracket(group_standings, best_thirds)

    for home, away in r32_pairs:
        stage_reached["r32"].add(home)
        stage_reached["r32"].add(away)

    # ── Knockout rounds ──────────────────────────────────────────────────────
    # Record who ENTERS each stage (not who wins), then simulate to get survivors.
    def play_round(pairs, stage_key):
        """Play all matches in a round; record entrants; return winners."""
        winners = []
        for home, away in pairs:
            stage_reached[stage_key].add(home)
            stage_reached[stage_key].add(away)
            winners.append(knockout_winner(model, home, away, elo_dict, rng))
        return winners

    r16_winners = play_round(r32_pairs, "r32")      # r32 entrants already set above
    # Overwrite r32 with only the 32 who actually entered R32
    # (already correct from the pair loop above)

    qf_winners = play_round(
        list(zip(r16_winners[0::2], r16_winners[1::2])), "r16"
    )
    sf_winners = play_round(
        list(zip(qf_winners[0::2], qf_winners[1::2])), "qf"
    )
    finalists = play_round(
        list(zip(sf_winners[0::2], sf_winners[1::2])), "sf"
    )
    # Record finalists
    for t in finalists:
        stage_reached["final"].add(t)
    # Simulate the final
    champion = knockout_winner(model, finalists[0], finalists[1], elo_dict, rng)
    stage_reached["champion"].add(champion)

    return {k: list(v) for k, v in stage_reached.items()}


def _build_r32_bracket(group_standings: dict[str, list[str]],
                        best_thirds: list[str]) -> list[tuple[str, str]]:
    """
    Build 16 R32 matchups from the 48-team field (32 teams total).
    - 12 group winners each face a runner-up or third-place team.
    - 4 group runners-up face other runners-up.
    - 8 best third-place teams fill the remaining slots.

    Layout (FIFA 2026 simplified bracket — exact draw TBD):
      Matches 0-7:  winner[i] vs runner-up from a different group
      Matches 8-11: winner[i+8] vs best_thirds[i]
      Matches 12-15: runner-up pairs (cross-bracket)

    Total: 16 matches, 32 distinct teams.
    """
    groups = sorted(group_standings.keys())  # A..L (12 groups)
    winners = [group_standings[g][0] for g in groups]      # 12 winners
    runners = [group_standings[g][1] for g in groups]       # 12 runners-up

    # Pair each winner against a runner-up from a non-adjacent group
    # Winners 0-7 (groups A-H) vs runners 8-11 + 4 thirds
    # Winners 8-11 (groups I-L) vs best thirds
    # Remaining runners (0-7 range) pair among themselves for 4 more matches
    # Assignment: 12 winners + 12 runners + 8 thirds = 32 teams, 16 matches.
    # Winners 0-7 each face a runner from the opposite half of the bracket.
    # Winners 8-11 each face a third-place team.
    # Runners 0-3 each face a third-place team.
    # Runners 4-7 face winners 4-7 (already covered above).
    # This gives 8 + 4 + 4 = 16 matches, all 32 teams distinct.
    #   winners[0..7] vs runners[4..11]   → 8 matches
    #   winners[8..11] vs thirds[0..3]    → 4 matches
    #   runners[0..3] vs thirds[4..7]     → 4 matches

    pairs: list[tuple[str, str]] = []
    for i in range(8):
        pairs.append((winners[i], runners[i + 4]))
    for i in range(4):
        third = best_thirds[i] if i < len(best_thirds) else runners[i]
        pairs.append((winners[8 + i], third))
    for i in range(4):
        third = best_thirds[4 + i] if (4 + i) < len(best_thirds) else winners[i]
        pairs.append((runners[i], third))

    return pairs  # 16 matches, 32 distinct teams


# ── Monte Carlo aggregation ───────────────────────────────────────────────────

def run_simulations(model, elo_dict: dict[str, float],
                    n: int = config.N_SIMULATIONS,
                    seed: Optional[int] = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Run n tournament simulations. Returns DataFrame with columns:
    team, group_advance, r32, r16, qf, sf, final, champion
    (as probabilities 0–1).
    """
    rng = np.random.default_rng(seed)

    teams = all_teams()
    counts: dict[str, dict[str, int]] = {
        t: {s: 0 for s in ["group", "r32", "r16", "qf", "sf", "final", "champion"]}
        for t in teams
    }

    for sim_i in range(n):
        result = simulate_tournament(model, elo_dict, rng)
        for stage, reached in result.items():
            for team in reached:
                if team in counts:
                    counts[team][stage] += 1

    rows = []
    for team in teams:
        c = counts[team]
        rows.append({
            "team": team,
            "group_advance": c["r32"] / n,   # made it out of groups (into R32)
            "r32": c["r32"] / n,
            "r16": c["r16"] / n,
            "qf": c["qf"] / n,
            "sf": c["sf"] / n,
            "final": c["final"] / n,
            "champion": c["champion"] / n,
        })

    df = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
    return df
