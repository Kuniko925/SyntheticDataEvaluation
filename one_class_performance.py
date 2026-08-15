import config
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main(models: list[str], seeds: list[int]) -> None:
    results_dir = Path(config.PROJECT_ROOT) / "results"
    rows = []

    for model in models:
        for seed in seeds:
            result_path = results_dir / f"{model}_ONE_{seed}.csv"

            if not result_path.exists():
                print(f"File not found: {result_path}")
                continue

            result_df = pd.read_csv(result_path)

            required_columns = {"label", "preds"}
            missing_columns = required_columns - set(result_df.columns)

            if missing_columns:
                raise ValueError(
                    f"{result_path} is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            y_true = result_df["label"]
            y_pred = result_df["preds"]

            labels = sorted(set(y_true.unique()) | set(y_pred.unique()))

            for label in labels:
                mask = y_true == label

                label_accuracy = (y_pred[mask] == label).mean()

                y_true_binary = (y_true == label).astype(int)
                y_pred_binary = (y_pred == label).astype(int)

                label_f1 = f1_score(
                    y_true_binary,
                    y_pred_binary,
                    zero_division=0,
                )

                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "label": label,
                        "accuracy": label_accuracy,
                        "f1": label_f1,
                    }
                )

    if not rows:
        raise RuntimeError("No valid result files were found.")

    raw_df = pd.DataFrame(rows).sort_values(
        ["model", "label", "seed"]
    )

    summary_df = (
        raw_df.groupby(["model", "label"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
    )

    summary_df["accuracy"] = summary_df.apply(
        lambda r: f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}",
        axis=1,
    )

    summary_df["f1"] = summary_df.apply(
        lambda r: f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}",
        axis=1,
    )

    summary_df = summary_df[["model", "label", "accuracy", "f1"]]
    summary_path = results_dir / "one_class_metrics_by_label_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nPer-seed results:")
    print(raw_df)

    print("\nSummary by label:")
    print(summary_df)


if __name__ == "__main__":
    models = ["ResNet50", "MobileNetV2", "ViT16"]
    seeds = [12, 123, 1234]

    main(models=models, seeds=seeds)