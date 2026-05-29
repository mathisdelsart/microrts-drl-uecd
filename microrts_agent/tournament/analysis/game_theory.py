"""
Game-Theoretic Evaluation Metrics for Tournament Analysis

Computes advanced rankings from a pairwise win-rate matrix:
  - Nash Averaging
  - Alpha-Rank
  - Condorcet Winner & Copeland Score
  - Average and Worst-Case Regret

All public functions take a winrate_matrix (n x n numpy array where
entry [i,j] is the win rate of agent i against agent j, in [0,1]).
"""

from typing import Optional

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Win-rate matrix construction
# ---------------------------------------------------------------------------


def build_winrate_matrix(games, ai_names: list[str]) -> np.ndarray:
    """
    Build pairwise win-rate matrix from game results.

    Args:
        games: List of GameData with ai1_name, ai2_name, winner attributes
        ai_names: Ordered list of AI names

    Returns:
        n x n matrix  winrate[i,j] = win rate of ai_names[i] vs ai_names[j]
    """
    n = len(ai_names)
    wins = np.zeros((n, n))
    counts = np.zeros((n, n))

    ai_idx = {name: i for i, name in enumerate(ai_names)}

    for game in games:
        i = ai_idx[game.ai1_name]
        j = ai_idx[game.ai2_name]
        counts[i, j] += 1
        counts[j, i] += 1
        if game.winner == 0:
            wins[i, j] += 1
            # wins[j, i] += 0
        elif game.winner == 1:
            wins[j, i] += 1
            # wins[i, j] += 0
        else:
            wins[i, j] += 0.5
            wins[j, i] += 0.5

    with np.errstate(invalid="ignore"):
        winrate = np.where(counts > 0, wins / counts, 0.5)
    np.fill_diagonal(winrate, 0.5)

    return winrate


# ---------------------------------------------------------------------------
# Condorcet Winner & Copeland Score
# ---------------------------------------------------------------------------


def condorcet_winner(winrate_matrix: np.ndarray, ai_names: list[str]) -> Optional[str]:
    """
    Condorcet winner: the agent that beats ALL others pairwise (>50%).
    Returns the name, or None if no such agent exists (intransitive cycle).
    """
    n = winrate_matrix.shape[0]
    for i in range(n):
        if all(winrate_matrix[i, j] > 0.5 for j in range(n) if j != i):
            return ai_names[i]
    return None


def copeland_scores(winrate_matrix: np.ndarray) -> np.ndarray:
    """
    Copeland score: #opponents beaten minus #opponents that beat us.
    Range [-(n-1), +(n-1)].  A Condorcet winner scores +(n-1).
    """
    n = winrate_matrix.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if winrate_matrix[i, j] > 0.5:
                scores[i] += 1
            elif winrate_matrix[i, j] < 0.5:
                scores[i] -= 1
    return scores


# ---------------------------------------------------------------------------
# Regret Metrics
# ---------------------------------------------------------------------------


def regret_metrics(winrate_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average and worst-case regret for each agent.

        regret(i, j) = max_k WR(k,j) - WR(i,j)

    Returns:
        avg_regret    : mean regret across opponents  (lower = better generalist)
        worst_regret  : max regret across opponents   (lower = fewer weaknesses)
        worst_matchup : index of the opponent causing worst regret
    """
    n = winrate_matrix.shape[0]
    best_vs = winrate_matrix.max(axis=0)  # best achievable WR vs each opponent

    avg_regret = np.zeros(n)
    worst_regret = np.zeros(n)
    worst_matchup = np.zeros(n, dtype=int)

    for i in range(n):
        regrets = best_vs - winrate_matrix[i, :]
        regrets[i] = 0  # exclude self-play

        mask = np.ones(n, dtype=bool)
        mask[i] = False

        avg_regret[i] = regrets[mask].mean()
        worst_regret[i] = regrets[mask].max()
        worst_matchup[i] = np.where(mask, regrets, -1).argmax()

    return avg_regret, worst_regret, worst_matchup


# ---------------------------------------------------------------------------
# Nash Averaging
# ---------------------------------------------------------------------------


def nash_averaging(winrate_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nash equilibrium of the symmetric zero-sum meta-game
    (Balduzzi et al., 2018 — "Re-evaluating Evaluation").

    Treats the tournament as a game: each "player" picks an agent from the
    pool. The payoff is the win rate centered at 0.5 (so it's zero-sum).
    We find the maximin mixed strategy — the optimal probability distribution
    over agents that an adversary would use.

    More principled than raw win rate because it weights opponents by their
    strategic importance: beating an agent everyone beats is worth little,
    beating an agent the Nash mixture relies on is worth a lot.

    Args:
        winrate_matrix : n x n pairwise win rates (entry [i,j] = WR of i vs j)

    Returns:
        nash_weights : probability distribution over agents (length n).
                       High weight = agent is strategically important / hard
                       to counter. These are the agents the optimal opponent
                       would play most often.
        nash_scores  : expected payoff of each agent against the Nash mixture
                       (length n). Positive = beats the optimal strategy,
                       negative = loses to it. Used for ranking.
    """
    n = winrate_matrix.shape[0]

    # Center winrates to make the game zero-sum:
    # A[i,j] = +0.2 means i beats j 70% of the time (advantage for i)
    # A[j,i] = -0.2 (symmetric disadvantage for j)
    A = winrate_matrix - 0.5

    # --- Solve the maximin LP ---
    # We want:  max v  s.t.  for all opponent j: p^T A e_j >= v
    #           with sum(p) = 1, p >= 0
    # i.e. find the mix p that maximizes the worst-case expected payoff v.
    #
    # scipy.linprog minimizes, so rewrite as:
    #   min -v  s.t.  -A^T p + v*1 <= 0  (one constraint per opponent j)

    # Decision variables: [p_0, p_1, ..., p_{n-1}, v]
    # Objective: minimize -v (= maximize v)
    c = np.zeros(n + 1)
    c[-1] = -1.0

    # Inequality constraints: -A^T p + v <= 0  for each opponent j
    # i.e. for each row j: sum_i(-A[i,j] * p_i) + v <= 0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)

    # Equality constraint: sum(p) = 1 (valid probability distribution)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p_i >= 0 (probabilities), v is unconstrained
    bounds = [(0, None)] * n + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        # Extract the Nash mixture (clamp tiny negatives from solver noise)
        weights = np.maximum(result.x[:n], 0)
        weights /= weights.sum()

        # Score each agent: expected payoff when facing the Nash mixture
        # scores[i] = sum_j A[i,j] * weights[j]
        scores = A @ weights
        return weights, scores

    # Fallback: if LP fails, assume uniform (shouldn't happen in practice)
    return np.ones(n) / n, np.zeros(n)


# ---------------------------------------------------------------------------
# Alpha-Rank
# ---------------------------------------------------------------------------


def alpha_rank(winrate_matrix: np.ndarray, alpha: float = 0.02, m: int = 50) -> np.ndarray:
    """
    Alpha-Rank scores via the Fermi fixation process on a finite population
    (Omidshafiei et al., 2019).

    Models evolutionary dynamics: a population of m individuals all play the
    same strategy. A single mutant appears playing a different strategy.
    Depending on relative fitness, the mutant either takes over (fixation)
    or dies out. Strategies that survive longest get higher scores.

    Always yields a unique ranking (unlike Nash, which can have multiple
    equilibria).

    Args:
        winrate_matrix : n x n pairwise win rates (entry [i,j] = WR of i vs j)
        alpha : selection intensity (higher = more deterministic, the stronger
                strategy almost always wins; lower = more random drift).
                Default 100 = strong selection (standard in the paper).
        m     : population size for the Moran process. Must be > n.
                Default 50 is the paper's standard. Larger m = slower but
                smoother results.

    Returns:
        pi : stationary distribution (length n). pi[i] = fraction of time
             strategy i dominates the population at equilibrium.
             Higher = better agent.
    """
    n = winrate_matrix.shape[0]
    payoff = winrate_matrix

    # --- Fixation probabilities ---
    # fixation[i,j] = probability that a SINGLE mutant playing strategy i,
    # inserted into a population of (m-1) individuals all playing strategy j,
    # eventually takes over the entire population (all m play i).
    # Computed via the Moran process with Fermi selection.
    fixation = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # We compute rho(i,j) = 1 / (1 + sum_{k=1}^{m-1} prod_{l=1}^{k} exp(-alpha*(f_i - f_j)))
            # using log-sum-exp for numerical stability (avoids overflow from exp(large)).
            # logaddexp(a, b) = log(exp(a) + exp(b)) without computing exp explicitly.
            log_terms = -np.inf  # log(0) — empty sum
            running_sum = 0.0  # accumulates the log of the product

            for k in range(1, m):
                # k = current number of mutants (strategy i) in the population
                # m-k = number of residents (strategy j)

                # f_j = fitness of resident j: avg payoff against current population
                #   plays vs (k-1) other j's  +  (m-k) copies of i
                f_j = ((k - 1) * payoff[j, j] + (m - k) * payoff[j, i]) / (m - 1)

                # f_i = fitness of mutant i: avg payoff against current population
                #   plays vs k copies of j  +  (m-k-1) other i's
                f_i = (k * payoff[i, j] + (m - k - 1) * payoff[i, i]) / (m - 1)

                # Accumulate -alpha * (f_j - f_i) in log space
                # If mutant i is stronger (f_i > f_j), this term is positive
                # → fixation probability increases
                running_sum += -alpha * (f_j - f_i)
                log_terms = np.logaddexp(log_terms, running_sum)

            # rho = 1 / (1 + sum_of_exp_terms) = exp(-log(1 + sum_of_exp_terms))
            fixation[i, j] = np.exp(-np.logaddexp(0.0, log_terms))

    # --- Transition matrix (Markov chain over monomorphic states) ---
    # Each state = "entire population plays strategy i"
    # T[i,j] = probability of going from "all play i" to "all play j"
    #         = (1/(n-1)) * fixation[j,i]
    #   because: a random mutant j appears (uniform over n-1 others),
    #   then it fixates with probability fixation[j,i].
    # T[i,i] = probability of staying = 1 - sum of leaving probabilities.
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                T[i, j] = fixation[i, j] / (n - 1)
        T[i, i] = 1.0 - T[i, :].sum()

    # Clamp negatives from floating-point errors and re-normalize rows
    T = np.maximum(T, 0)
    T /= T.sum(axis=1, keepdims=True)

    # --- Stationary distribution ---
    # Find eigenvector of T^T with eigenvalue 1 (guaranteed to exist for
    # a stochastic matrix). This is the long-run fraction of time spent
    # in each monomorphic state.
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.abs(np.real(eigenvectors[:, idx]))
    pi /= pi.sum()

    return pi
