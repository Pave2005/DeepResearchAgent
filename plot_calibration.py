import json
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_ece_and_plot(json_path: str = "results/benchmark_results_20261002_110235.json", n_bins: int = 5):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Файл '{json_path}' не найден в текущей директории!")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        raise ValueError("В переданном JSON-файле отсутствует список 'results'")

    def calculate_bins(subset):
        confs = np.array([r["system_confidence"] for r in subset])
        precs = np.array([r["citation_precision"] for r in subset])
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        ece = 0.0
        bin_centers = []
        bin_accs = []
        bin_counts = []

        for i in range(n_bins):
            b_low, b_high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (confs >= b_low) & (confs <= b_high)
            else:
                mask = (confs >= b_low) & (confs < b_high)

            count = np.sum(mask)
            bin_counts.append(count)
            bin_centers.append((b_low + b_high) / 2.0)

            if count > 0:
                acc = np.mean(precs[mask])
                conf = np.mean(confs[mask])
                ece += (count / len(subset)) * np.abs(acc - conf)
                bin_accs.append(acc)
            else:
                bin_accs.append(0.0)

        return ece, bin_edges, bin_centers, bin_accs, bin_counts

    standard_subset = [r for r in results if not r.get("is_trap", False)]
    ece_std, edges_std, centers_std, accs_std, counts_std = calculate_bins(standard_subset)
    ece_all, edges_all, centers_all, accs_all, counts_all = calculate_bins(results)

    print("=" * 60)
    print("ИТОГИ КАЛИБРОВКИ СИСТЕМЫ (ECE АНАЛИЗ)")
    print("=" * 60)
    print(f"Всего тестов в файле:                     {len(results)}")
    print(f"  • ECE на стандартных тестах (n={len(standard_subset)}):    {ece_std * 100:.2f}%")
    print(f"  • ECE на всей выборке с ловушками (n={len(results)}):  {ece_all * 100:.2f}%")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Идеальная калибровка (y = x)")
    bars1 = ax1.bar(
        edges_std[:-1], accs_std, width=1.0 / n_bins, align="edge",
        alpha=0.6, color="#1f77b4", edgecolor="black", label="Эмпирическая точность (Precision)"
    )
    ax1.set_title(f"Стандартные тесты (n={len(standard_subset)})\nECE = {ece_std * 100:.2f}%", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Системная уверенность (Confidence)", fontsize=11)
    ax1.set_ylabel("Точность цитирования (Precision)", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    for idx, (bar, count) in enumerate(zip(bars1, counts_std)):
        if count > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Идеальная калибровка (y = x)")
    bars2 = ax2.bar(
        edges_all[:-1], accs_all, width=1.0 / n_bins, align="edge",
        alpha=0.6, color="#ff7f0e", edgecolor="black", label="Эмпирическая точность (Precision)"
    )
    ax2.set_title(f"Полная выборка с ловушками (n={len(results)})\nECE = {ece_all * 100:.2f}%", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Системная уверенность (Confidence)", fontsize=11)
    ax2.set_ylabel("Точность цитирования (Precision)", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    for idx, (bar, count) in enumerate(zip(bars2, counts_all)):
        if count > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    plt.tight_layout()
    output_img = "results/calibration_reliability_diagram.png"
    plt.savefig(output_img, dpi=300)
    print(f"График успешно сохранен: {output_img}\n")

if __name__ == "__main__":
    compute_ece_and_plot()
