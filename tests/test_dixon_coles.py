"""Tests for Dixon-Coles model correctness."""
import numpy as np
import pytest
from dixon_coles import DixonColesModel, _tau


def test_tau_zero_zero():
    """τ(0,0) = 1 - μ₁μ₂ρ"""
    mu1 = np.array([1.5])
    mu2 = np.array([1.2])
    rho = 0.1
    result = _tau(np.array([0]), np.array([0]), mu1, mu2, rho)
    expected = 1 - 1.5 * 1.2 * 0.1
    assert abs(result[0] - expected) < 1e-10


def test_tau_one_one():
    """τ(1,1) = 1 - ρ"""
    result = _tau(np.array([1]), np.array([1]), np.array([1.0]), np.array([1.0]), 0.2)
    assert abs(result[0] - 0.8) < 1e-10


def test_tau_neutral_cells():
    """τ for (2,0) and other high-score cells should be 1."""
    for x, y in [(2, 0), (0, 2), (2, 2), (3, 1)]:
        result = _tau(np.array([x]), np.array([y]),
                      np.array([1.5]), np.array([1.2]), 0.1)
        assert abs(result[0] - 1.0) < 1e-10, f"τ({x},{y}) should be 1"


def test_score_matrix_sums_to_one():
    """Score probability matrix should sum to ~1."""
    m = DixonColesModel()
    m._fitted = True
    mat = m.score_matrix(1.5, 1.2)
    assert abs(mat.sum() - 1.0) < 1e-6


def test_score_matrix_probabilities_non_negative():
    m = DixonColesModel()
    m._fitted = True
    mat = m.score_matrix(2.0, 0.8)
    assert (mat >= 0).all()


def test_predict_probabilities_sum_to_one():
    """P(home win) + P(draw) + P(away win) ≈ 1."""
    m = DixonColesModel()
    m._fitted = True
    m._teams = ["A", "B"]
    m.attack = {"A": 0.2, "B": -0.1}
    m.defense = {"A": -0.1, "B": 0.1}
    m.home_adv = 0.15
    m.rho = -0.05

    pred = m.predict("A", "B", neutral=True)
    total = pred["p_home_win"] + pred["p_draw"] + pred["p_away_win"]
    assert abs(total - 1.0) < 1e-5


def test_predict_favourite_has_higher_win_prob():
    """A strong team should have higher win probability than a weak team."""
    m = DixonColesModel()
    m._fitted = True
    m._teams = ["Strong", "Weak"]
    m.attack = {"Strong": 0.8, "Weak": -0.8}
    m.defense = {"Strong": -0.4, "Weak": 0.4}
    m.home_adv = 0.0
    m.rho = -0.05

    pred = m.predict("Strong", "Weak", neutral=True)
    assert pred["p_home_win"] > pred["p_away_win"]


def test_sample_score_returns_valid_ints():
    m = DixonColesModel()
    m._fitted = True
    m._teams = ["X", "Y"]
    m.attack = {"X": 0.1, "Y": -0.1}
    m.defense = {"X": -0.1, "Y": 0.1}
    m.home_adv = 0.0
    m.rho = 0.0

    rng = np.random.default_rng(42)
    for _ in range(20):
        hg, ag = m.sample_score("X", "Y", rng=rng)
        assert isinstance(hg, int) and hg >= 0
        assert isinstance(ag, int) and ag >= 0


def test_rho_negative_increases_low_score_probs():
    """Negative rho should inflate 0-0 probability above independent Poisson."""
    m = DixonColesModel()
    m._fitted = True

    mat_rho0 = m.score_matrix(1.2, 1.1)   # rho defaults to 0
    m.rho = -0.1
    mat_rho_neg = m.score_matrix(1.2, 1.1)

    # With negative rho, 0-0 should be more likely than independent Poisson
    assert mat_rho_neg[0, 0] > mat_rho0[0, 0]
