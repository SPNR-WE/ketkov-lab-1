import os
import time
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.stochastic.stochastic import StochasticSolver
from src.stochastic.evaluate import StochasticEvaluator
from src.general import SetCoveringProblem

np.random.seed(42)


def draw_plots(exp_name: str, config: dict, results: dict):
    # создаем папку для сохранения графиков
    os.makedirs("report/images", exist_ok=True)

    # собираем фигуру с тремя графиками в ряд
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # формируем общий заголовок с параметрами
    title = (f"Эксперимент: {exp_name} | "
             f"m={config['m']}, density={config['density']}, "
             f"alpha={config['alpha']}, k={config['k_samples']}")
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # первый график: время работы алгоритмов
    ax1.plot(results["n"], results["rn_time"], marker='o', label='Risk-Neutral')
    ax1.plot(results["n"], results["ra_time"], marker='s', label='Risk-Averse')
    ax1.set_xlabel("Количество множеств (n)")
    ax1.set_ylabel("Среднее время решения (с)")
    ax1.set_title("Время решения")
    ax1.grid(True)
    ax1.legend()

    # второй график: ошибки несоответствия (mismatch error)
    ax2.errorbar(results["n"], results["mismatch_cvar_mean"], yerr=results["mismatch_cvar_std"],
                 fmt='-o', capsize=4, label='Ошибка CVaR (RN в RA)')
    ax2.errorbar(results["n"], results["mismatch_mean_mean"], yerr=results["mismatch_mean_std"],
                 fmt='-s', capsize=4, label='Ошибка Ср. Затрат (RA в RN)')
    ax2.set_xlabel("Количество множеств (n)")
    ax2.set_ylabel("Mismatch Gap")
    ax2.set_title("Ошибка несоответствия моделей")
    ax2.grid(True)
    ax2.legend()

    # третий график: оптимистичное смещение (bias)
    ax3.errorbar(results["n"], results["rn_bias_mean"], yerr=results["rn_bias_std"],
                 fmt='-o', capsize=4, label='RN: SAA Bias (Mean)')
    ax3.errorbar(results["n"], results["ra_bias_mean"], yerr=results["ra_bias_std"],
                 fmt='-s', capsize=4, label='RA: SAA Bias (CVaR)')
    ax3.set_xlabel("Количество множеств (n)")
    ax3.set_ylabel("Out-of-sample - In-sample")
    ax3.set_title("Оптимистичное смещение (SAA Bias)")
    ax3.grid(True)
    ax3.legend()

    # выравниваем и сохраняем картинку
    plt.tight_layout()
    plt.savefig(f"report/images/{exp_name}.png")
    plt.close()


def run_experiment(exp_name: str, config: dict) -> dict:
    m = config["m"]
    density = config["density"]
    k_samples = config["k_samples"]
    problems = config["problems"]
    alpha = config["alpha"]
    n_values = config["n_values"]

    results = defaultdict(list)
    total_iterations = len(n_values) * problems

    with tqdm(total=total_iterations, desc=f"Эксперимент: {exp_name}", unit="it") as pbar:
        for n in n_values:
            rn_times, ra_times = [], []
            rn_means, ra_means = [], []
            rn_cvars, ra_cvars = [], []
            rn_biases, ra_biases = [], []

            for _ in range(problems):
                # передаем seed=None, чтобы генератор продолжал глобальную случайную последовательность
                problem = SetCoveringProblem(m=m, n=n, density=density, seed=None)
                C = problem.generate_samples(k=k_samples)
                solver = StochasticSolver(problem.A)

                start = time.time()
                x_rn, in_obj_rn = solver.solve_risk_neutral(C)
                rn_times.append(time.time() - start)

                start = time.time()
                x_ra, in_obj_ra = solver.solve_risk_averse(C, alpha=alpha)
                ra_times.append(time.time() - start)

                # создаем оценщик (он внутри себя сгенерирует 1000 тестовых сценариев и заморозит их)
                evaluator = StochasticEvaluator(
                    c_mean=problem.c_mean,
                    c_std=problem.c_std,
                    alpha=alpha,
                    n_eval=1000
                )

                # тестируем оба решения на одних и тех же данных!
                mean_rn, cvar_rn = evaluator.evaluate(x_rn)
                mean_ra, cvar_ra = evaluator.evaluate(x_ra)

                rn_means.append(mean_rn)
                ra_means.append(mean_ra)
                rn_cvars.append(cvar_rn)
                ra_cvars.append(cvar_ra)

                rn_biases.append(mean_rn - in_obj_rn)
                ra_biases.append(cvar_ra - in_obj_ra)

                pbar.update(1)

            # усредняем результаты для текущего n
            results["n"].append(n)
            results["rn_time"].append(np.mean(rn_times))
            results["ra_time"].append(np.mean(ra_times))

            # считаем mismatch error по cvar
            mismatch_cvar_gap = np.array(rn_cvars) - np.array(ra_cvars)
            results["mismatch_cvar_mean"].append(np.mean(mismatch_cvar_gap))
            results["mismatch_cvar_std"].append(np.std(mismatch_cvar_gap))

            # считаем mismatch error по среднему
            mismatch_mean_gap = np.array(ra_means) - np.array(rn_means)
            results["mismatch_mean_mean"].append(np.mean(mismatch_mean_gap))
            results["mismatch_mean_std"].append(np.std(mismatch_mean_gap))

            # сохраняем bias
            results["rn_bias_mean"].append(np.mean(rn_biases))
            results["rn_bias_std"].append(np.std(rn_biases))
            results["ra_bias_mean"].append(np.mean(ra_biases))
            results["ra_bias_std"].append(np.std(ra_biases))

            # выводим подсказку с текущим размером матрицы
            pbar.set_postfix({"завершён n": n})

    # рисуем графики после прохода всех n
    draw_plots(exp_name, config, results)
    return results


def main():
    # подтягиваем путь к конфигурации
    config_path = os.path.join(os.path.dirname(__file__), "stochastic_exp.yml")

    # парсим yaml-конфиг
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    # хранилище для всех таблиц
    all_results = {}

    # запускаем эксперименты один за другим
    for exp_name, exp_config in full_config.get("experiments", {}).items():
        all_results[exp_name] = (exp_config, run_experiment(exp_name, exp_config))

    # --- СОХРАНЕНИЕ И ВЫВОД ТАБЛИЦЫ ---

    # создаем папку report, если ее вдруг нет
    report_file = "report/tables/stochastic-table.md"

    # формируем строки таблицы
    table_lines = [
        "## Результаты экспериментов (Стохастическое программирование)\n",
        "| Эксперимент (Параметры) | n | Время RN (с) | Время RA (с) | Mismatch CVaR | Mismatch Mean | Bias RN | Bias RA |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for exp_name, (config, results) in all_results.items():
        # формируем жирный подзаголовок с параметрами
        params_str = f"**{exp_name}**<br>m={config['m']}, d={config['density']}, $\\alpha$={config['alpha']}"

        for i in range(len(results["n"])):
            # вставляем подзаголовок только в первую строку блока
            col1 = params_str if i == 0 else ""

            line = (f"| {col1} | {results['n'][i]} | {results['rn_time'][i]:.4f} | {results['ra_time'][i]:.4f} | "
                    f"{results['mismatch_cvar_mean'][i]:.4f} ± {results['mismatch_cvar_std'][i]:.2f} | "
                    f"{results['mismatch_mean_mean'][i]:.4f} ± {results['mismatch_mean_std'][i]:.2f} | "
                    f"{results['rn_bias_mean'][i]:.4f} | {results['ra_bias_mean'][i]:.4f} |")
            table_lines.append(line)

    # склеиваем все строки через перенос
    final_markdown = "\n".join(table_lines)

    # печатаем в консоль
    print("\n\n--- ОБЩАЯ СВОДНАЯ ТАБЛИЦА ---\n")
    print(final_markdown)

    # записываем в файл
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"\nТаблица успешно сохранена в {report_file}")

if __name__ == "__main__":
    main()
