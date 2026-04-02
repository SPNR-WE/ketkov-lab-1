import time
from collections import defaultdict

import numpy as np
import pulp

M = 20
K_SAMPLES = 30
PROBLEMS = 100
N_VALUES = list(range(5, 55, 5))
ALPHA = 0.9

np.random.seed(42)


def generate_problem(m, n):
    A = (np.random.rand(m, n) < 0.3).astype(int)

    for i in range(m):
        if A[i].sum() == 0:
            A[i, np.random.randint(n)] = 1

    return A


def sample_costs(n, k):
    return np.random.lognormal(mean=0.0, sigma=0.5, size=(k, n))


def solve_risk_neutral(A, C):
    k, n = C.shape
    m = A.shape[0]

    model = pulp.LpProblem("SetCover_RN", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")

    model += (1.0 / k) * pulp.lpSum(C[s][j] * x[j] for s in range(k) for j in range(n))

    for i in range(m):
        model += pulp.lpSum(A[i][j] * x[j] for j in range(n)) >= 1

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    return np.array([pulp.value(x[j]) for j in range(n)])


def solve_risk_averse(A, C, alpha):
    k, n = C.shape
    m = A.shape[0]

    model = pulp.LpProblem("SetCover_CVaR", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    eta = pulp.LpVariable("eta")
    z = pulp.LpVariable.dicts("z", range(k), lowBound=0)

    model += eta + (1.0 / ((1 - alpha) * k)) * pulp.lpSum(z[s] for s in range(k))

    for s in range(k):
        model += z[s] >= pulp.lpSum(C[s][j] * x[j] for j in range(n)) - eta

    for i in range(m):
        model += pulp.lpSum(A[i][j] * x[j] for j in range(n)) >= 1

    model.solve(pulp.PULP_CBC_CMD(msg=0))
    return np.array([pulp.value(x[j]) for j in range(n)])


def evaluate_solution(x, alpha, n_eval=1000):
    n = len(x)
    costs = np.random.lognormal(mean=0.0, sigma=0.5, size=(n_eval, n))

    vals = costs @ x
    mean = np.mean(vals)

    threshold = np.quantile(vals, alpha)
    cvar = vals[vals >= threshold].mean()

    return mean, cvar


def run_experiment():
    results = defaultdict(list)

    for n in N_VALUES:
        risk_neutral_vals, risk_averse_vals = [], []
        risk_neutral_cvar_vals, risk_averse_cvar_vals = [], []
        risk_neutral_times, risk_averse_times = [], []

        for _ in range(PROBLEMS):
            A = generate_problem(M, n)
            C = sample_costs(n, K_SAMPLES)

            start = time.time()
            x_rn = solve_risk_neutral(A, C)
            risk_neutral_times.append(time.time() - start)

            start = time.time()
            x_ra = solve_risk_averse(A, C, ALPHA)
            risk_averse_times.append(time.time() - start)

            mean_rn, cvar_rn = evaluate_solution(x_rn, ALPHA)
            mean_ra, cvar_ra = evaluate_solution(x_ra, ALPHA)

            risk_neutral_vals.append(mean_rn)
            risk_averse_vals.append(mean_ra)
            risk_neutral_cvar_vals.append(cvar_rn)
            risk_averse_cvar_vals.append(cvar_ra)

        results["n"].append(n)

        results["rn_mean"].append(
            (np.mean(risk_neutral_vals), np.std(risk_neutral_vals))
        )
        results["ra_mean"].append((np.mean(risk_averse_vals), np.std(risk_averse_vals)))

        results["rn_cvar"].append(
            (np.mean(risk_neutral_cvar_vals), np.std(risk_neutral_cvar_vals))
        )
        results["ra_cvar"].append(
            (np.mean(risk_averse_cvar_vals), np.std(risk_averse_cvar_vals))
        )

        results["rn_time"].append(np.mean(risk_neutral_times))
        results["ra_time"].append(np.mean(risk_averse_times))

    return results


def report_results(results):
    for i, n in enumerate(results["n"]):
        rn_cvar = results["rn_cvar"][i][0]
        ra_cvar = results["ra_cvar"][i][0]

        rn_mean = results["rn_mean"][i][0]
        ra_mean = results["ra_mean"][i][0]

        print(f"n={n}")
        print(
            f"RN used for CVaR: gap = {rn_cvar - ra_cvar:.4f}, RA used for mean: gap = {ra_mean - rn_mean:.4f}"
        )
        print(f"RN={results['rn_time'][i]:.4f}s, RA={results['ra_time'][i]:.4f}s")


def main():
    results = run_experiment()
    report_results(results)


if __name__ == "__main__":
    main()
