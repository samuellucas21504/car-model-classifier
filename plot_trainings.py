from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

import training_config

def load_history(csv_path: Path):
    """
    Lê um CSV (no formato salvo pelo save_history_to_csv) e retorna:
    - epochs: lista de ints
    - metrics: dict[str, list[float]]
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        epochs = []
        metrics = {}

        for row in reader:
            epochs.append(int(row["epoch"]))
            for key, value in row.items():
                if key == "epoch":
                    continue
                try:
                    v = float(value)
                except ValueError:
                    # ignora colunas não numéricas
                    continue
                metrics.setdefault(key, []).append(v)

    return epochs, metrics


def plot_version_comparison(version_dir: Path):
    """
    Para uma pasta de versão (ex: trainings/1.0):

    - Lê baseline.csv e finetune.csv
    - Descobre as métricas em comum
    - Gera gráfico(s) baseline vs finetune para cada métrica
    """
    baseline_csv = version_dir / "baseline.csv"
    finetune_csv = version_dir / "finetune.csv"

    if not (baseline_csv.exists() and finetune_csv.exists()):
        return

    print(f"[INFO] Gerando gráficos para versão: {version_dir.name}")

    base_epochs, base_metrics = load_history(baseline_csv)
    fine_epochs, fine_metrics = load_history(finetune_csv)

    common_metrics = sorted(set(base_metrics.keys()) & set(fine_metrics.keys()))
    if not common_metrics:
        print(f"[WARN] Nenhuma métrica em comum em {version_dir}")
        return

    n_metrics = len(common_metrics)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    fig.suptitle(f"Versão {version_dir.name} - baseline vs finetune", fontsize=16)

    for idx, metric in enumerate(common_metrics):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        ax.plot(base_epochs, base_metrics[metric], label="baseline")
        ax.plot(fine_epochs, fine_metrics[metric], label="finetune")

        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    # Esconde subplots sobrando
    for idx in range(len(common_metrics), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = version_dir / "plots_baseline_vs_finetune.png"
    fig.savefig(output_path, dpi=150)
    print(f"[OK] Gráfico por versão salvo em: {output_path}")

    # Mostra interativamente (opcional)
    # plt.show()
    plt.close(fig)


# ==========================================================
# Comparativo GLOBAL de todas as versões
# ==========================================================
def plot_all_versions_comparison(global_histories):
    """
    global_histories: lista de dicts, cada item:
        {
          "version": str,
          "baseline": {"epochs": [...], "metrics": {...}},
          "finetune": {"epochs": [...], "metrics": {...}},
        }

    Gera, para cada métrica comum em TODAS as versões, um gráfico
    com todas as versões no mesmo plot:

        versão-baseline e versão-finetune

    Salvando na raiz de trainings como:
        plots_ALL_versions_<metric>.png
    """
    if not global_histories:
        print("[WARN] Sem versões para comparativo global.")
        return

    # Descobre métricas comuns a todas as versões (baseline e finetune)
    common_metrics = None
    for item in global_histories:
        base_metrics = set(item["baseline"]["metrics"].keys())
        fine_metrics = set(item["finetune"]["metrics"].keys())
        metrics_this_version = base_metrics & fine_metrics

        if common_metrics is None:
            common_metrics = metrics_this_version
        else:
            common_metrics &= metrics_this_version

    if not common_metrics:
        print("[WARN] Nenhuma métrica em comum em todas as versões (baseline+finetune).")
        return

    common_metrics = sorted(common_metrics)

    # TRAININGS_ROOT para salvar os gráficos globais
    root: Path = training_config.TRAININGS_ROOT

    print("[INFO] Métricas em comum para comparativo global:", common_metrics)

    for metric in common_metrics:
        # Um gráfico por métrica, com todas as versões
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Comparativo global - {metric}", fontsize=16)

        for item in global_histories:
            version = item["version"]

            base_epochs = item["baseline"]["epochs"]
            base_values = item["baseline"]["metrics"][metric]
            fine_epochs = item["finetune"]["epochs"]
            fine_values = item["finetune"]["metrics"][metric]

            # Linhas: versão-baseline e versão-finetune
            ax.plot(
                base_epochs,
                base_values,
                label=f"{version}-baseline",
                linestyle="-",
            )
            ax.plot(
                fine_epochs,
                fine_values,
                label=f"{version}-finetune",
                linestyle="--",
            )

        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = root / f"plots_ALL_versions_{metric}.png"
        fig.savefig(output_path, dpi=150)
        print(f"[OK] Gráfico global salvo em: {output_path}")

        # Mostra interativamente (opcional)
        # plt.show()
        plt.close(fig)


# ==========================================================
# main
# ==========================================================
def main():
    root: Path = training_config.TRAININGS_ROOT  # ex: Path("trainings")
    if not root.exists():
        print(f"[ERRO] Diretório raiz de trainings não existe: {root}")
        return

    print(f"[INFO] Procurando versões em: {root.resolve()}")

    # Todas as subpastas de trainings/
    version_dirs = [d for d in root.iterdir() if d.is_dir()]
    version_dirs = sorted(version_dirs, key=lambda p: p.name)

    if not version_dirs:
        print("[WARN] Nenhuma subpasta de versão encontrada.")
        return

    # Para alimentar o comparativo global
    global_histories = []

    for vdir in version_dirs:
        baseline_csv = vdir / "baseline.csv"
        finetune_csv = vdir / "finetune.csv"

        if not (baseline_csv.exists() and finetune_csv.exists()):
            print(f"[SKIP] {vdir.name} não tem baseline.csv e finetune.csv; pulando.")
            continue

        # 1) gráfico por versão (já existia)
        plot_version_comparison(vdir)

        # 2) guarda dados para gráfico global
        base_epochs, base_metrics = load_history(baseline_csv)
        fine_epochs, fine_metrics = load_history(finetune_csv)

        global_histories.append(
            {
                "version": vdir.name,
                "baseline": {"epochs": base_epochs, "metrics": base_metrics},
                "finetune": {"epochs": fine_epochs, "metrics": fine_metrics},
            }
        )

    # 3) gráfico global de todas as versões no mesmo plot
    plot_all_versions_comparison(global_histories)


if __name__ == "__main__":
    main()
