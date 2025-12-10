from pathlib import Path
import csv

VERSION = "1.0"

TRAININGS_ROOT = Path("trainings")


def get_version_dir() -> Path:
    """
    Retorna o diretório da versão atual, criando se não existir.
    Ex: trainings/0.3/
    """
    version_dir = TRAININGS_ROOT / VERSION
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def save_history_to_csv(history, stage: str):
    """
    Salva o history do Keras em trainings/<VERSION>/<stage>.csv

    stage: 'baseline', 'finetune', 'extra', etc.
    """
    version_dir = get_version_dir()
    csv_path = version_dir / f"{stage}.csv"

    logs = history.history
    n_epochs = len(next(iter(logs.values())))
    epochs = range(1, n_epochs + 1)

    fieldnames = ["epoch"] + list(logs.keys())

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, epoch in enumerate(epochs):
            row = {"epoch": epoch}
            for metric_name, values in logs.items():
                row[metric_name] = float(values[i])
            writer.writerow(row)

    print(f"[LOG] Histórico salvo em: {csv_path}")
