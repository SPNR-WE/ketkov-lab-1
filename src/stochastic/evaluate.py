import numpy as np

class StochasticEvaluator:
    def __init__(self, c_mean: np.ndarray, c_std: np.ndarray, alpha: float = 0.9, n_eval: int = 1000):
        self.alpha = alpha

        # генерируем тестовую выборку один раз при инициализации
        costs = np.random.normal(loc=c_mean, scale=c_std, size=(n_eval, len(c_mean)))
        self.test_costs = np.abs(costs)

    def evaluate(self, x: np.ndarray) -> tuple[float, float]:
        # считаем стоимости для заданного решения x на замороженной выборке
        vals = self.test_costs @ x

        # вычисляем истинное среднее
        mean_val = float(np.mean(vals))

        # вычисляем истинный cvar (среднее по худшим сценариям)
        threshold = np.quantile(vals, self.alpha)
        cvar_val = float(vals[vals >= threshold].mean())

        return mean_val, cvar_val
