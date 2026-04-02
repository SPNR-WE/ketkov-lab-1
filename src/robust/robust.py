import gurobipy as gp
from gurobipy import GRB
import numpy as np

class RobustSolver:
    def __init__(self, A_matrix: np.ndarray, env: gp.Env):
        self.A = A_matrix
        self.m, self.n = A_matrix.shape
        self.env = env

    def solve_robust(self, c_bar: np.ndarray, c_hat: np.ndarray, gamma: float, warm_start: dict = None) -> tuple[np.ndarray, float, dict]:
        model = gp.Model(env=self.env)
        model.Params.OutputFlag = 0  # Отключаем логирование солвера
        model.Params.Threads = 1

        # Бинарные переменные выбора множеств
        x = model.addVars(self.n, vtype=GRB.BINARY, name="x")

        # Двойственные непрерывные неотрицательные переменные
        mu = model.addVar(lb=0.0, name="mu")
        nu = model.addVars(self.n, lb=0.0, name="nu")

        # Целевая функция робастной одноуровневой задачи
        model.setObjective(
            gp.quicksum(float(c_bar[j]) * x[j] for j in range(self.n)) +
            float(gamma) * mu +
            gp.quicksum(nu[j] for j in range(self.n)),
            GRB.MINIMIZE
        )

        # Ограничения покрытия
        for i in range(self.m):
            model.addConstr(gp.quicksum(self.A[i, j] * x[j] for j in range(self.n)) >= 1)

        # Робастные ограничения
        for j in range(self.n):
            model.addConstr(mu + nu[j] >= float(c_hat[j]) * x[j])

        # Применяем warm start, если он передан (существенно ускоряет цикл по Gamma)
        if warm_start is not None:
            for j in range(self.n):
                x[j].Start = float(warm_start["x"][j])
                nu[j].Start = float(warm_start["nu"][j])
            mu.Start = float(warm_start["mu"])

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Robust model failed, status {model.Status}")

        # Извлекаем решение
        x_sol = np.array([round(x[j].X) for j in range(self.n)], dtype=int)
        mu_sol = float(mu.X)
        nu_sol = np.array([nu[j].X for j in range(self.n)], dtype=float)

        return x_sol, float(model.ObjVal), {"x": x_sol, "mu": mu_sol, "nu": nu_sol}

    def solve_deterministic(self, c_true: np.ndarray) -> tuple[np.ndarray, float]:
        model = gp.Model(env=self.env)
        model.Params.OutputFlag = 0
        model.Params.Threads = 1

        x = model.addVars(self.n, vtype=GRB.BINARY, name="x")

        model.setObjective(gp.quicksum(float(c_true[j]) * x[j] for j in range(self.n)), GRB.MINIMIZE)

        for i in range(self.m):
            model.addConstr(gp.quicksum(self.A[i, j] * x[j] for j in range(self.n)) >= 1)

        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Deterministic model failed, status {model.Status}")

        x_sol = np.array([round(x[j].X) for j in range(self.n)], dtype=int)
        return x_sol, float(model.ObjVal)
