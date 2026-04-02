import numpy as np

class SetCoveringProblem:
    def __init__(self, m: int, n: int, density: float = 0.2, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.m = m # число покрываемых элементов
        self.n = n # число множеств
        self.density = density # плотность покрываемых элементов (чем выше, тем больше элементов покрывает каждое множество)
        self.A = None # матрица a_{ij} с множествами
        self.c_mean = None # матожидание генерируемых значений
        self.c_std = None # стандартное отклонение генерируемых значений

        self._generate_instance()

    def _generate_instance(self):
        # генерируем матрицу покрытия a_{ij}
        self.A = np.random.choice(
            [0, 1],
            size=(self.m, self.n),
            p=[1 - self.density, self.density]
        )

        # гарантируем допустимость: если строка пустая, ставим случайную 1
        for i in range(self.m):
            if np.sum(self.A[i, :]) == 0:
                self.A[i, np.random.randint(0, self.n)] = 1

        # задаем распределение стоимостей
        self.c_mean = np.random.uniform(10.0, 50.0, size=self.n)
        self.c_std = np.random.uniform(1.0, 10.0, size=self.n)

    def generate_samples(self, k: int) -> np.ndarray:
        # генерируем k сценариев стоимостей (матрица k x n)
        samples = np.random.normal(loc=self.c_mean, scale=self.c_std, size=(k, self.n))
        samples = np.abs(samples) # в распределении могут быть отрицательные веса, обработаем их через модуль
        return samples

    def f(self, x: np.ndarray, c: np.ndarray) -> float:
        return float(np.dot(c, x))

    def f_saa(self, x: np.ndarray, c_samples: np.ndarray) -> tuple[float, float]:
        costs = np.dot(c_samples, x)
        return float(np.mean(costs)), float(np.std(costs))

    def is_covered(self, x: np.ndarray) -> bool:
        return bool(np.all(np.dot(self.A, x) >= 1))
