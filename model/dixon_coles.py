"""
Dixon-Coles (1997) bivariate Poisson model for football scoreline prediction.

Implements:
- Team-specific attack/defense parameters (log-linear parameterization)
- Global home-advantage multiplier
- Low-score correction factor rho for cells (0,0), (1,0), (0,1), (1,1)
- Exponential time-decay weighting (xi ~ 0.0065/day)

Fitted via MLE using scipy.optimize.minimize (L-BFGS-B).
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tau(x, y, lam_h, lam_a, rho):
    """Dixon-Coles correction factor for low-scoring cells."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    elif x == 0 and y == 1:
        return 1 + lam_h * rho
    elif x == 1 and y == 0:
        return 1 + lam_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def scoreline_matrix(lam_h, lam_a, rho, max_goals=9):
    """
    Full P(X=x, Y=y) matrix for x, y in [0, max_goals).
    Incorporates the Dixon-Coles tau correction.
    """
    mat = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            p_h = poisson.pmf(x, lam_h)
            p_a = poisson.pmf(y, lam_a)
            t = _tau(x, y, lam_h, lam_a, rho)
            mat[x, y] = p_h * p_a * t
    # Re-normalize for truncation at max_goals
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def marginalize_goals(mat):
    """
    From an (N x N) scoreline matrix, compute over/under probabilities.
    Returns dict with prob_under25/over25, under35/over35, under45/over45.
    """
    n = mat.shape[0]
    p_under25 = 0.0
    p_under35 = 0.0
    p_under45 = 0.0
    for x in range(n):
        for y in range(n):
            if x + y <= 2:
                p_under25 += mat[x, y]
            if x + y <= 3:
                p_under35 += mat[x, y]
            if x + y <= 4:
                p_under45 += mat[x, y]
    return {
        "prob_under25": float(p_under25),
        "prob_over25":  float(1 - p_under25),
        "prob_under35": float(p_under35),
        "prob_over35":  float(1 - p_under35),
        "prob_under45": float(p_under45),
        "prob_over45":  float(1 - p_under45),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model.

    Parameterization (log-linear):
        log(lambda_home) = home_adv + attack[home] + defense[away]
        log(lambda_away) =            attack[away] + defense[home]

    Identifiability constraint: attack[0] = 0  (reference team, alphabetically first).
    """

    def __init__(self, xi=0.0065):
        self.xi = xi               # time-decay rate (per day)
        self.teams_ = None
        self.team_to_idx_ = None
        self.attack_ = None        # length n
        self.defense_ = None       # length n
        self.home_adv_ = None      # scalar
        self.rho_ = None           # scalar

    # ------------------------------------------------------------------ fit

    def _neg_log_likelihood(self, params, h_idx, a_idx, gh, ga, weights, n_teams):
        """Vectorized negative log-likelihood."""
        # Unpack
        att = np.zeros(n_teams)
        att[1:] = params[:n_teams - 1]
        def_ = params[n_teams - 1: 2 * n_teams - 1]
        home = params[2 * n_teams - 1]
        rho = np.clip(params[2 * n_teams], -0.99, 0.99)

        # Expected goals
        lam_h = np.exp(np.clip(home + att[h_idx] + def_[a_idx], -5, 3))
        lam_a = np.exp(np.clip(att[a_idx] + def_[h_idx], -5, 3))

        # Poisson log-pmf:  x*log(lam) - lam - log(x!)
        log_ph = gh * np.log(lam_h + 1e-10) - lam_h - gammaln(gh + 1)
        log_pa = ga * np.log(lam_a + 1e-10) - lam_a - gammaln(ga + 1)

        # Tau correction (vectorized)
        tau = np.ones_like(gh, dtype=float)
        m00 = (gh == 0) & (ga == 0)
        m10 = (gh == 1) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m11 = (gh == 1) & (ga == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m11] = 1 - rho
        tau = np.maximum(tau, 1e-10)

        ll = weights * (log_ph + log_pa + np.log(tau))

        # Light L2 regularization to keep params reasonable
        reg = 0.001 * (np.sum(att ** 2) + np.sum(def_ ** 2))

        return -ll.sum() + reg

    def fit(self, home_teams, away_teams, goals_home, goals_away,
            match_dates=None, reference_date=None):
        """
        Fit via MLE on historical match results.

        Parameters
        ----------
        home_teams, away_teams : array-like of str
        goals_home, goals_away : array-like of int
        match_dates : array-like of datetime (optional, for time decay)
        reference_date : datetime (optional, default=max(match_dates))
        """
        self.teams_ = sorted(set(home_teams) | set(away_teams))
        self.team_to_idx_ = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        h_idx = np.array([self.team_to_idx_[t] for t in home_teams])
        a_idx = np.array([self.team_to_idx_[t] for t in away_teams])
        gh = np.asarray(goals_home, dtype=float)
        ga = np.asarray(goals_away, dtype=float)

        # Time-decay weights
        if match_dates is not None:
            dates = pd.to_datetime(match_dates)
            ref = pd.Timestamp(reference_date) if reference_date else dates.max()
            days_ago = (ref - dates).days.values.astype(float) if hasattr(ref - dates, 'days') else ((ref - dates) / pd.Timedelta(days=1)).values.astype(float)
            weights = np.exp(-self.xi * days_ago)
        else:
            weights = np.ones(len(gh))

        # Initial guess: (n-1) attack + n defense + home + rho = 2n free params
        n_params = 2 * n + 1
        x0 = np.zeros(n_params)
        x0[2 * n - 1] = 0.25   # initial home advantage
        x0[2 * n] = -0.05      # initial rho

        # Bounds
        bounds = (
            [(-3, 3)] * (n - 1) +   # attack[1:]
            [(-3, 3)] * n +          # defense[0:]
            [(-1, 2)] +             # home_adv
            [(-0.99, 0.99)]         # rho
        )

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(h_idx, a_idx, gh, ga, weights, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )

        # Store
        self.attack_ = np.zeros(n)
        self.attack_[1:] = result.x[:n - 1]
        self.defense_ = result.x[n - 1: 2 * n - 1]
        self.home_adv_ = result.x[2 * n - 1]
        self.rho_ = np.clip(result.x[2 * n], -0.99, 0.99)

        return self

    # -------------------------------------------------------------- predict

    def predict_match(self, home_team, away_team, max_goals=9):
        """
        Predict a single match.

        Returns
        -------
        dict with keys:
            lambda_home, lambda_away, scoreline_matrix,
            prob_over25, prob_under25, prob_over35, prob_under35
        """
        h_idx = self.team_to_idx_.get(home_team)
        a_idx = self.team_to_idx_.get(away_team)

        # Fallback for unknown teams: use average attack/defense
        avg_att = float(np.mean(self.attack_))
        avg_def = float(np.mean(self.defense_))

        att_h = self.attack_[h_idx] if h_idx is not None else avg_att
        def_h = self.defense_[h_idx] if h_idx is not None else avg_def
        att_a = self.attack_[a_idx] if a_idx is not None else avg_att
        def_a = self.defense_[a_idx] if a_idx is not None else avg_def

        lam_h = np.exp(self.home_adv_ + att_h + def_a)
        lam_a = np.exp(att_a + def_h)

        mat = scoreline_matrix(lam_h, lam_a, self.rho_, max_goals)
        probs = marginalize_goals(mat)

        return {
            "lambda_home": float(lam_h),
            "lambda_away": float(lam_a),
            "scoreline_matrix": mat,
            **probs,
        }

    def predict_batch(self, home_teams, away_teams):
        """Predict multiple matches. Returns list of dicts (without scoreline matrices)."""
        results = []
        for h, a in zip(home_teams, away_teams):
            pred = self.predict_match(h, a)
            del pred["scoreline_matrix"]  # drop large matrix for batch use
            results.append(pred)
        return results

    # --------------------------------------------------------- save / load

    def save(self, path):
        data = {
            "xi": self.xi,
            "teams": self.teams_,
            "attack": self.attack_.tolist(),
            "defense": self.defense_.tolist(),
            "home_adv": float(self.home_adv_),
            "rho": float(self.rho_),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        m = cls(xi=data["xi"])
        m.teams_ = data["teams"]
        m.team_to_idx_ = {t: i for i, t in enumerate(m.teams_)}
        m.attack_ = np.array(data["attack"])
        m.defense_ = np.array(data["defense"])
        m.home_adv_ = data["home_adv"]
        m.rho_ = data["rho"]
        return m

    # -------------------------------------------------------------- debug

    def print_params(self, top_n=10):
        """Print strongest/weakest attack and defense parameters."""
        idx = np.argsort(self.attack_)[::-1]
        print(f"\nHome advantage: {self.home_adv_:.4f}  (multiplier: {np.exp(self.home_adv_):.3f}x)")
        print(f"Rho (low-score correction): {self.rho_:.4f}")
        print(f"\nTop {top_n} attack:")
        for i in idx[:top_n]:
            print(f"  {self.attack_[i]:+.4f}  {self.teams_[i]}")
        print(f"\nBottom {top_n} attack:")
        for i in idx[-top_n:]:
            print(f"  {self.attack_[i]:+.4f}  {self.teams_[i]}")
        idx_d = np.argsort(self.defense_)
        print(f"\nStrongest defense (lowest = fewest goals conceded):")
        for i in idx_d[:top_n]:
            print(f"  {self.defense_[i]:+.4f}  {self.teams_[i]}")
