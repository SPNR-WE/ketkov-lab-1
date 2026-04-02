import os
import time
from collections import defaultdict

import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gurobipy as gp

from src.robust.robust import RobustSolver
from src.robust.evaluator import RobustEvaluator
from src.general import SetCoveringProblem

def setup_gurobi_env() -> gp.Env:
    env = gp.Env(empty=True)
    # Параметры лицензии WLS (подставьте свои из блокнота)
    env.setParam('WLSAccessID', 'c883e76c-6b5d-4a16-9dc8-dd44f341af22')
    env.setParam('WLSSecret', '1721a792-2ef0-4ae2-a7c7-993621b5fcca')
    env.setParam('LicenseID', 2799306)
    env.start()
    return env

def draw_plots(exp_name: str, config: dict, results: dict):
    os.makedirs("report/robust_images", exist_ok=True)

    n_values = results["n"]
    avg_times = results["avg_time_fixed_gamma"]

    # 1. График времени работы
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, avg_times, marker="o", color="blue", linewidth=2)
    plt.xlabel("Количество множеств (n)")
    plt.ylabel(f"Среднее время (с) при $\\Gamma$={config['gamma_fixed']}")
    plt.title(f"Эксперимент: {exp_name} | Зависимость времени от n")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"report/robust_images/{exp_name}_time.png", dpi=200)
    plt.close()

    # 2. Графики In-sample и Out-of-sample для самого большого исследованного n
    last_n = n_values[-1]
    gammas = np.arange(last_n + 1)

    ins_mean = results["ins_mean_curves"][-1]
    ins_std = results["ins_std_curves"][-1]
    oos_mean = results["oos_mean_curves"][-1]
    oos_std = results["oos_std_curves"][-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Эксперимент: {exp_name} | Метрики для n={last_n}", fontsize=14, fontweight='bold')

    # In-sample subplot
    ax1.errorbar(gammas, ins_mean, yerr=ins_std, marker="o", capsize=3, color="green")
    ax1.set_xlabel(r"Бюджет неопределенности $\Gamma$")
    ax1.set_ylabel("In-sample performance")
    ax1.set_title("In-sample vs Gamma")
    ax1.grid(True, alpha=0.3)

    # Out-of-sample subplot
    ax2.errorbar(gammas, oos_mean, yerr=oos_std, marker="s", capsize=3, color="red")
    ax2.set_xlabel(r"Бюджет неопределенности $\Gamma$")
    ax2.set_ylabel("Out-of-sample performance")
    ax2.set_title("Out-of-sample vs Gamma")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"report/robust_images/{exp_name}_curves_n_{last_n}.png", dpi=200)
    plt.close()


def run_experiment(exp_name: str, config: dict, env: gp.Env) -> dict:
    n_values = config["n_values"]
    m_fixed = config["m_fixed"]
    density = config["density"]
    n_instances = config["n_instances"]
    n_oos = config["n_oos_scenarios"]
    gamma_fixed = config["gamma_fixed"]

    results = defaultdict(list)

    for n in tqdm(n_values, desc=f"Эксперимент {exp_name} (по n)"):
        # Используем существующий класс задачи для получения матрицы покрытия A
        problem = SetCoveringProblem(m=m_fixed, n=n, density=density, seed=42+n)
        A = problem.A

        # Генерируем фиксированные отклонения c_hat для текущего n
        c_hat = np.random.randint(config["c_hat_low"], config["c_hat_high"] + 1, size=n).astype(float)

        solver = RobustSolver(A, env)

        insample_ratio = np.zeros((n_instances, n + 1))
        oos_ratio = np.zeros((n_instances, n + 1))
        time_fixed_gamma = []

        for inst in tqdm(range(n_instances), desc=f"Инстансы для n={n}", leave=False):
            # Для каждого инстанса генерируем новые номинальные стоимости c_bar
            c_bar = np.random.randint(config["c_bar_low"], config["c_bar_high"] + 1, size=n).astype(float)
            gamma_true = int(config["gamma_true_frac"] * n)

            # Инициализация оценщика автоматически подготовит эталонные оптимумы (Out-of-sample)
            evaluator = RobustEvaluator(c_bar, c_hat, gamma_true, n_oos, solver, seed=1000+inst)

            warm_start = None
            for gamma in range(n + 1):
                start_t = time.time()
                x_sol, obj_val, warm_start = solver.solve_robust(c_bar, c_hat, gamma, warm_start)
                t_elapsed = time.time() - start_t

                # Замеряем время только для зафиксированного Gamma (поведение как в блокноте)
                if gamma == min(gamma_fixed, n):
                    time_fixed_gamma.append(t_elapsed)

                # Сохраняем оценки
                ins, oos = evaluator.evaluate(x_sol, obj_val)
                insample_ratio[inst, gamma] = ins
                oos_ratio[inst, gamma] = oos

        # Агрегация результатов по всем 100 инстансам
        ins_mean = insample_ratio.mean(axis=0)
        ins_std = insample_ratio.std(axis=0, ddof=1)
        oos_mean = oos_ratio.mean(axis=0)
        oos_std = oos_ratio.std(axis=0, ddof=1)

        # Находим "лучший" Gamma, при котором модель показала себя лучше всего на OOS
        opt_gamma = int(np.argmin(oos_mean))

        results["n"].append(n)
        results["avg_time_fixed_gamma"].append(np.mean(time_fixed_gamma))
        results["opt_gamma"].append(opt_gamma)
        results["min_oos_mean"].append(oos_mean[opt_gamma])
        results["ins_at_opt_gamma"].append(ins_mean[opt_gamma])

        # Сохраняем полные кривые для графиков
        results["ins_mean_curves"].append(ins_mean)
        results["ins_std_curves"].append(ins_std)
        results["oos_mean_curves"].append(oos_mean)
        results["oos_std_curves"].append(oos_std)

    draw_plots(exp_name, config, results)
    return results


def main():
    env = setup_gurobi_env()

    config_path = os.path.join(os.path.dirname(__file__), "robust_exp.yml")
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    all_results = {}

    for exp_name, exp_config in full_config.get("experiments", {}).items():
        all_results[exp_name] = (exp_config, run_experiment(exp_name, exp_config, env))

    # --- СОХРАНЕНИЕ И ВЫВОД ТАБЛИЦЫ ---
    os.makedirs("report/tables", exist_ok=True)
    report_file = "report/tables/robust-table.md"

    table_lines = [
        "## Результаты экспериментов (Робастное программирование)\n",
        "| Эксперимент | n | Время при фикс. $\\Gamma$ (с) | Опт. $\\Gamma$ (по OOS) | Min OOS Ratio | In-sample при опт. $\\Gamma$ |",
        "|---|---|---|---|---|---|"
    ]

    for exp_name, (config, results) in all_results.items():
        params_str = f"**{exp_name}**<br>m={config['m_fixed']}, d={config['density']}, $\\Gamma_{{true}}$={config['gamma_true_frac']}n"

        for i in range(len(results["n"])):
            col1 = params_str if i == 0 else ""
            line = (f"| {col1} | {results['n'][i]} | {results['avg_time_fixed_gamma'][i]:.4f} | "
                    f"{results['opt_gamma'][i]} | {results['min_oos_mean'][i]:.4f} | "
                    f"{results['ins_at_opt_gamma'][i]:.4f} |")
            table_lines.append(line)

    final_markdown = "\n".join(table_lines)
    print("\n\n--- ОБЩАЯ СВОДНАЯ ТАБЛИЦА ---\n")
    print(final_markdown)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"\nТаблица успешно сохранена в {report_file}")

if __name__ == "__main__":
    main()
