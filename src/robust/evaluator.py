import numpy as np
from src.robust.robust import RobustSolver

class RobustEvaluator:
    def __init__(self, c_bar: np.ndarray, c_hat: np.ndarray, gamma_true: int, n_oos: int, solver: RobustSolver, seed: int = None):
        self.c_bar = c_bar
        self.c_hat = c_hat
        self.gamma_true = gamma_true
        self.n_oos = n_oos
        self.solver = solver

        self.rng = np.random.default_rng(seed)

        # Генерируем "истинный" in-sample сценарий
        self.c_true_in = self._generate_budgeted()

        # Генерируем набор out-of-sample сценариев
        self.c_true_oos = np.array([self._generate_budgeted() for _ in range(n_oos)])

        # Заранее решаем детерминированные задачи для OOS, чтобы найти эталонные оптимумы (знаменатель для out-of-sample)
        self.true_opt_costs = np.array([
            self.solver.solve_deterministic(c)[1] for c in self.c_true_oos
        ])

    def _generate_budgeted(self) -> np.ndarray:
        """Генерирует настоящие стоимости в рамках бюджетного множества неопределенности."""
        n = len(self.c_bar)
        xi = np.zeros(n)

        # Выбираем случайные индексы, которые "портятся"
        idx = self.rng.choice(n, size=min(self.gamma_true, n), replace=False)
        xi[idx] = 1.0

        return self.c_bar + self.c_hat * xi

    def evaluate(self, x_sol: np.ndarray, robust_est_cost: float) -> tuple[float, float]:
        # In-sample performance (Оценка робастной модели / Реальность на обучающей выборке)
        true_cost_in = float(self.c_true_in @ x_sol)
        insample_ratio = robust_est_cost / true_cost_in if true_cost_in > 0 else 1.0

        # Out-of-sample performance (Стоимость нашего решения / Абсолютный оптимум на тестовых данных)
        our_cost_oos = self.c_true_oos @ x_sol
        oos_ratio = float(np.mean(our_cost_oos / self.true_opt_costs))

        return insample_ratio, oos_ratio
