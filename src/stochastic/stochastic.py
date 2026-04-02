import pulp
import numpy as np

class StochasticSolver:
    def __init__(self, A_matrix: np.ndarray):
        # сохраняем матрицу покрытия и ее размерности
        self.A = A_matrix
        self.m, self.n = A_matrix.shape

    def solve_risk_neutral(self, cost_samples: np.ndarray, msg: int = 0) -> tuple[np.ndarray, float]:
        k, n = cost_samples.shape

        model = pulp.LpProblem("SetCover_RiskNeutral", pulp.LpMinimize)

        # задаем бинарные переменные решения
        x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)

        # минимизируем мат. ожидание затрат
        model += (1.0 / k) * pulp.lpSum(
            cost_samples[s, j] * x[j] for s in range(k) for j in range(n)
        )

        # добавляем ограничения на покрытие всех элементов
        for i in range(self.m):
            model += pulp.lpSum(self.A[i, j] * x[j] for j in range(n)) >= 1

        # запускаем солвер
        model.solve(pulp.PULP_CBC_CMD(msg=msg))

        # возвращаем вектор x и in-sample значение функции
        return np.array([pulp.value(x[j]) for j in range(n)]), float(pulp.value(model.objective))

    def solve_risk_averse(self, cost_samples: np.ndarray, alpha: float = 0.9, msg: int = 0) -> tuple[np.ndarray, float]:
        k, n = cost_samples.shape

        model = pulp.LpProblem("SetCover_RiskAverse_CVaR", pulp.LpMinimize)

        # задаем переменные (бинарные для множеств, непрерывные для cvar)
        x = pulp.LpVariable.dicts("x", range(n), cat=pulp.LpBinary)
        t = pulp.LpVariable("t")
        z = pulp.LpVariable.dicts("z", range(k), lowBound=0)

        # минимизируем cvar
        model += t + (1.0 / ((1.0 - alpha) * k)) * pulp.lpSum(z[s] for s in range(k))

        # ограничения для вычисления превышения порога
        for s in range(k):
            model += z[s] >= pulp.lpSum(cost_samples[s, j] * x[j] for j in range(n)) - t

        # ограничения на покрытие всех элементов
        for i in range(self.m):
            model += pulp.lpSum(self.A[i, j] * x[j] for j in range(n)) >= 1

        # запускаем солвер
        model.solve(pulp.PULP_CBC_CMD(msg=msg))

        # возвращаем вектор x и in-sample cvar
        return np.array([pulp.value(x[j]) for j in range(n)]), float(pulp.value(model.objective))
