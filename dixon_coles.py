"""
Dixon-Coles bivariate Poisson model with:
  - Per-team attack/defense strength parameters
  - Home advantage (suppressed at neutral venues)
  - Exponential time-decay weighting
  - Low-score rho correction (Dixon-Coles 1997)
  - Optional Elo-based shrinkage prior
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import config


# ── Low-score correction ──────────────────────────────────────────────────────

def _tau(x: np.ndarray, y: np.ndarray, mu1: np.ndarray,
         mu2: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles rho adjustment factor τ(x, y, μ₁, μ₂, ρ)."""
    result = np.ones_like(x, dtype=float)
    mask_00 = (x == 0) & (y == 0)
    mask_10 = (x == 1) & (y == 0)
    mask_01 = (x == 0) & (y == 1)
    mask_11 = (x == 1) & (y == 1)
    result[mask_00] = 1 - mu1[mask_00] * mu2[mask_00] * rho
    result[mask_10] = 1 + mu2[mask_10] * rho
    result[mask_01] = 1 + mu1[mask_01] * rho
    result[mask_11] = 1 - rho
    return result


# ── Model class ───────────────────────────────────────────────────────────────

class DixonColesModel:
    """
    Fitted Dixon-Coles model.  Call fit() to estimate parameters, then
    predict() for scoreline probabilities.
    """

    def __init__(self, elo_dict: Optional[dict[str, float]] = None):
        self.elo_dict = elo_dict or {}
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self._teams: list[str] = []
        self._fitted = False

    # ── Parameter packing / unpacking ────────────────────────────────────────

    def _pack(self, attack: dict, defense: dict, home_adv: float,
              rho: float) -> np.ndarray:
        atk = np.array([attack[t] for t in self._teams])
        dfc = np.array([defense[t] for t in self._teams])
        return np.concatenate([atk, dfc, [home_adv, rho]])

    def _unpack(self, params: np.ndarray):
        n = len(self._teams)
        attack = dict(zip(self._teams, params[:n]))
        defense = dict(zip(self._teams, params[n:2*n]))
        home_adv = params[2*n]
        rho = params[2*n + 1]
        return attack, defense, home_adv, rho

    # ── Negative log-likelihood ───────────────────────────────────────────────

    def _neg_log_likelihood(self, params: np.ndarray, home_idx: np.ndarray,
                            away_idx: np.ndarray, hg: np.ndarray,
                            ag: np.ndarray, neutral: np.ndarray,
                            weights: np.ndarray) -> float:
        n = len(self._teams)
        atk = params[:n]
        dfc = params[n:2*n]
        home_adv = params[2*n]
        rho = np.clip(params[2*n + 1], -0.99, 0.99)

        # Expected goals
        mu1 = np.exp(atk[home_idx] - dfc[away_idx] + home_adv * (~neutral).astype(float))
        mu2 = np.exp(atk[away_idx] - dfc[home_idx])

        # Poisson log-probs
        log_p1 = poisson.logpmf(hg, mu1)
        log_p2 = poisson.logpmf(ag, mu2)

        # Dixon-Coles correction
        tau = _tau(hg, ag, mu1, mu2, rho)
        tau = np.clip(tau, 1e-10, None)

        log_lik = weights * (log_p1 + log_p2 + np.log(tau))
        return -np.sum(log_lik)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, weights: np.ndarray,
            cutoff_date: Optional[str] = None) -> "DixonColesModel":
        """
        Fit on rows of df with columns [home, away, hgoals, agoals, neutral].
        weights: per-row importance weight (time-decay etc.).
        cutoff_date: if provided, only use data before this date (for backtesting).
        """
        if cutoff_date:
            mask = df["date"] < pd.Timestamp(cutoff_date)
            df = df[mask]
            weights = weights[mask.to_numpy()]

        # Keep only teams with ≥3 appearances
        team_counts = pd.concat([df["home"], df["away"]]).value_counts()
        valid_teams = team_counts[team_counts >= 3].index.tolist()
        mask = df["home"].isin(valid_teams) & df["away"].isin(valid_teams)
        df = df[mask].reset_index(drop=True)
        weights = weights[mask.to_numpy()]

        self._teams = sorted(set(df["home"]) | set(df["away"]))
        t2i = {t: i for i, t in enumerate(self._teams)}
        n = len(self._teams)

        home_idx = df["home"].map(t2i).to_numpy()
        away_idx = df["away"].map(t2i).to_numpy()
        hg = df["hgoals"].to_numpy()
        ag = df["agoals"].to_numpy()
        neutral = df["neutral"].to_numpy().astype(bool)

        # Initial params: zero attack/defense, small home adv, rho=0
        x0 = np.zeros(2*n + 2)
        x0[2*n] = 0.1   # home advantage
        x0[2*n+1] = -0.1  # rho

        # Constraint: sum of attack params = 0 (identifiability)
        constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]
        bounds = (
            [(-3, 3)] * n +      # attack
            [(-3, 3)] * n +      # defense (more positive = weaker)
            [(-1, 2)] +          # home advantage
            [(-0.99, 0.99)]      # rho
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._neg_log_likelihood,
                x0,
                args=(home_idx, away_idx, hg, ag, neutral, weights),
                method="L-BFGS-B",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 2000, "ftol": 1e-9},
            )

        self.attack, self.defense, self.home_adv, self.rho = self._unpack(result.x)

        # Elo shrinkage: blend fitted attack toward Elo-implied prior
        if self.elo_dict and config.ELO_BLEND_WEIGHT > 0:
            self._apply_elo_shrinkage()

        self._fitted = True
        print(f"Dixon-Coles fit: {len(self._teams)} teams, "
              f"home_adv={self.home_adv:.3f}, rho={self.rho:.3f}, "
              f"converged={result.success}")
        return self

    def _apply_elo_shrinkage(self):
        """Shrink team attack toward Elo-implied strength."""
        w = config.ELO_BLEND_WEIGHT
        mean_elo = np.mean(list(self.elo_dict.values())) if self.elo_dict else 1500.0
        for team in self._teams:
            elo = self.elo_dict.get(team, mean_elo)
            elo_implied = (elo - mean_elo) / 400.0  # rough log-scale
            self.attack[team] = (1 - w) * self.attack[team] + w * elo_implied

    # ── Predict ───────────────────────────────────────────────────────────────

    def expected_goals(self, home: str, away: str,
                       neutral: bool = True,
                       host_bump_home: float = 0.0,
                       host_bump_away: float = 0.0) -> tuple[float, float]:
        """Return (mu_home, mu_away) expected goals."""
        atk_h = self.attack.get(home, 0.0) + host_bump_home
        atk_a = self.attack.get(away, 0.0) + host_bump_away
        dfc_h = self.defense.get(home, 0.0)
        dfc_a = self.defense.get(away, 0.0)
        ha = 0.0 if neutral else self.home_adv
        mu_h = np.exp(atk_h - dfc_a + ha)
        mu_a = np.exp(atk_a - dfc_h)
        return float(mu_h), float(mu_a)

    def score_matrix(self, mu_h: float, mu_a: float,
                     max_goals: int = 8) -> np.ndarray:
        """
        (max_goals+1) × (max_goals+1) matrix of P(home=i, away=j).
        Applies Dixon-Coles rho correction for low scores.
        """
        goals = np.arange(max_goals + 1)
        p_h = poisson.pmf(goals, mu_h)
        p_a = poisson.pmf(goals, mu_a)
        matrix = np.outer(p_h, p_a)

        # Apply rho correction to 0-0, 1-0, 0-1, 1-1
        rho = self.rho
        for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            if i <= max_goals and j <= max_goals:
                t = _tau(
                    np.array([i]), np.array([j]),
                    np.array([mu_h]), np.array([mu_a]), rho
                )[0]
                matrix[i, j] *= t

        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()  # renormalize after correction
        return matrix

    def predict(self, home: str, away: str, neutral: bool = True,
                host_bump_home: float = 0.0,
                host_bump_away: float = 0.0) -> dict:
        """
        Return dict with keys:
          score_matrix, p_home_win, p_draw, p_away_win, mu_home, mu_away
        """
        assert self._fitted, "Call fit() first"
        mu_h, mu_a = self.expected_goals(home, away, neutral,
                                          host_bump_home, host_bump_away)
        mat = self.score_matrix(mu_h, mu_a)
        p_home = float(np.tril(mat, -1).sum())   # home score > away score
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())
        return {
            "score_matrix": mat,
            "p_home_win": p_home,
            "p_draw": p_draw,
            "p_away_win": p_away,
            "mu_home": mu_h,
            "mu_away": mu_a,
        }

    def sample_score(self, home: str, away: str, neutral: bool = True,
                     host_bump_home: float = 0.0,
                     host_bump_away: float = 0.0,
                     rng: Optional[np.random.Generator] = None) -> tuple[int, int]:
        """Sample a single scoreline from the score distribution."""
        if rng is None:
            rng = np.random.default_rng()
        mu_h, mu_a = self.expected_goals(home, away, neutral,
                                          host_bump_home, host_bump_away)
        mat = self.score_matrix(mu_h, mu_a)
        flat = mat.flatten()
        idx = rng.choice(len(flat), p=flat / flat.sum())
        r, c = divmod(idx, mat.shape[1])
        return int(r), int(c)


# ── Validation helpers ────────────────────────────────────────────────────────

def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Binary or 3-class log loss."""
    eps = 1e-15
    return -np.mean(outcomes * np.log(np.clip(probs, eps, 1 - eps)))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def backtest(df_full: pd.DataFrame, tournament: str,
             tournament_year: int,
             model: DixonColesModel) -> dict:
    """
    Train on data before tournament_year, evaluate on matches in tournament.
    Returns dict of metrics.
    """
    cutoff = f"{tournament_year}-01-01"
    weights_full = data_layer_weights(df_full)

    model_bt = DixonColesModel(elo_dict=model.elo_dict)
    model_bt.fit(df_full, weights_full, cutoff_date=cutoff)

    test = df_full[
        (df_full["date"] >= pd.Timestamp(cutoff)) &
        (df_full["tournament"].str.contains("FIFA World Cup", na=False)) &
        (~df_full["tournament"].str.contains("qualification", case=False, na=False))
    ].copy()

    records = []
    for _, row in test.iterrows():
        try:
            pred = model_bt.predict(row["home"], row["away"], neutral=True)
        except Exception:
            continue
        actual_h = row["hgoals"]
        actual_a = row["agoals"]
        if actual_h > actual_a:
            outcome = 0  # home win
        elif actual_h == actual_a:
            outcome = 1  # draw
        else:
            outcome = 2  # away win
        records.append({
            "p_home": pred["p_home_win"],
            "p_draw": pred["p_draw"],
            "p_away": pred["p_away_win"],
            "outcome": outcome,
        })

    if not records:
        return {"n": 0}

    rec_df = pd.DataFrame(records)
    probs = rec_df[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes_oh = np.eye(3)[rec_df["outcome"].to_numpy()]

    ll = -np.mean(
        np.log(np.clip(probs[np.arange(len(probs)), rec_df["outcome"].to_numpy()], 1e-15, 1))
    )
    bs = float(np.mean((probs - outcomes_oh) ** 2))
    return {
        "tournament": f"WC {tournament_year}",
        "n_matches": len(records),
        "log_loss": round(ll, 4),
        "brier_score": round(bs, 4),
    }


def data_layer_weights(df: pd.DataFrame) -> np.ndarray:
    """Convenience: compute weights using data_layer, avoiding circular import."""
    from data_layer import compute_weights
    return compute_weights(df)
