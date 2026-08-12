import config
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, accuracy_score


def build_training_df(result_dir, models, epochs, seeds):

    rows = []

    for model in models:
        for epoch in epochs:
            for seed in seeds:

                if epoch == 0:
                    csv_path = result_dir / f"{model}_FAKE1_REAL_{seed}.csv"
                else:
                    csv_path = result_dir / f"{model}_ADDITIONAL_{epoch}_{seed}.csv"

                df = pd.read_csv(csv_path)

                y_true = df["label"]
                y_pred = df["preds"]

                report = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0
                )

                for label_text, scores in report.items():

                    if not str(label_text).isdigit():
                        continue

                    class_id = int(label_text)

                    # class-wise accuracy: one-vs-rest
                    true_binary = (y_true == class_id)
                    pred_binary = (y_pred == class_id)

                    class_acc = accuracy_score(
                        true_binary,
                        pred_binary
                    )

                    rows.append({
                        "model": model,
                        "epoch": epoch,
                        "seed": seed,
                        "class_id": class_id,
                        "f1": scores["f1-score"],
                        "accuracy": class_acc,
                    })

    return pd.DataFrame(rows)


def build_real_df(result_dir, models, seeds):

    rows = []

    for model in models:
        for seed in seeds:

            csv_path = result_dir / f"{model}_REAL_REAL_{seed}.csv"
            df = pd.read_csv(csv_path)

            y_true = df["label"]
            y_pred = df["preds"]

            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            for label_text, scores in report.items():

                if not str(label_text).isdigit():
                    continue

                class_id = int(label_text)

                true_binary = (y_true == class_id)
                pred_binary = (y_pred == class_id)

                class_acc = accuracy_score(
                    true_binary,
                    pred_binary
                )

                rows.append({
                    "model": model,
                    "seed": seed,
                    "class_id": class_id,
                    "f1": scores["f1-score"],
                    "accuracy": class_acc,
                })

    return pd.DataFrame(rows)


def summarize_metric(df, metric, include_epoch=True):

    group_cols = ["model", "class_id"]

    if include_epoch:
        group_cols.append("epoch")

    summary = (
        df
        .groupby(group_cols, as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary = summary.rename(
        columns={
            "mean": f"{metric}_mean",
            "std": f"{metric}_std",
        }
    )

    return summary


def plot_metric(
    summary_df,
    real_summary_df,
    metric,
    models,
    classes,
    model_colors,
    save_path,
):

    metric_settings = {
        "f1": {
            "ylabel": "F1",
            "real_title": "REAL MEAN F1",
        },
        "accuracy": {
            "ylabel": "Accuracy",
            "real_title": "REAL MEAN ACC",
        },
    }

    settings = metric_settings[metric]

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 7),
        sharex=True,
        sharey=True
    )

    for idx, class_id in enumerate(classes):

        ax = axes[idx // 5, idx % 5]

        ax.set_title(
            config.label_to_class[class_id],
            fontsize=14
        )

        ax.set_ylim(0, 1)

        for model in models:

            dm = summary_df[
                (summary_df["model"] == model) &
                (summary_df["class_id"] == class_id)
            ].copy()

            dm = dm.sort_values("epoch")

            x = dm["epoch"].to_numpy()
            mean = dm[mean_col].to_numpy()
            std = dm[std_col].to_numpy()

            ax.plot(
                x,
                mean,
                color=model_colors[model],
                marker="o",
                linewidth=1.8,
                markersize=5,
            )

            ax.fill_between(
                x,
                np.clip(mean - std, 0, 1),
                np.clip(mean + std, 0, 1),
                color=model_colors[model],
                alpha=0.15,
            )

        # REAL mean
        real_sub = real_summary_df[
            real_summary_df["class_id"] == class_id
        ]

        real_text_lines = []

        for model in models:

            row = real_sub[
                real_sub["model"] == model
            ]

            if row.empty:
                continue

            value = row[mean_col].iloc[0]

            real_text_lines.append(
                f"{model}: {value:.3f}"
            )

        ax.text(
            0.98,
            0.03,
            settings["real_title"]
            + "\n"
            + "\n".join(real_text_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            )
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        if idx // 5 == 1:
            ax.set_xlabel(
                "Epochs",
                fontsize=13
            )

        if idx % 5 == 0:
            ax.set_ylabel(
                settings["ylabel"],
                fontsize=13
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=model_colors[model],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=model,
        )
        for model in models
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(models),
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0.07, 1, 1)
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("saved:", save_path)

def save_metric_summary_csv(
        training_df,
        real_df,
        result_dir,
        max_epoch=50,
):
    target_df = training_df[
        training_df["epoch"] <= max_epoch
        ].copy()

    summary = (
        target_df
        .groupby(
            ["model", "class_id", "epoch"],
            as_index=False
        )["f1"]
        .mean()
        .rename(columns={"f1": "f1_mean"})
    )

    summary["class_name"] = summary["class_id"].map(
        config.label_to_class
    )

    mean_table = summary.pivot(
        index=["model", "class_id", "class_name"],
        columns="epoch",
        values="f1_mean"
    )

    epoch_list = sorted(mean_table.columns.tolist())

    out_df = mean_table.reset_index()[
        ["model", "class_id", "class_name"]
    ].copy()

    # Epoch 0
    first_epoch = epoch_list[0]

    out_df[f"Epoch_{int(first_epoch)}_mean"] = (
        mean_table[first_epoch].to_numpy()
    )

    for prev_epoch, current_epoch in zip(
            epoch_list[:-1],
            epoch_list[1:]
    ):
        current_mean = mean_table[current_epoch]
        prev_mean = mean_table[prev_epoch]

        out_df[f"Epoch_{int(current_epoch)}_mean"] = (
            current_mean.to_numpy()
        )

        out_df[
            f"Improve_{int(prev_epoch)}_{int(current_epoch)}"
        ] = (
                current_mean - prev_mean
        ).to_numpy()

    real_summary = (
        real_df
        .groupby(
            ["model", "class_id"],
            as_index=False
        )["f1"]
        .mean()
        .rename(columns={"f1": "REAL_mean"})
    )

    out_df = out_df.merge(
        real_summary,
        on=["model", "class_id"],
        how="left"
    )


    numeric_cols = out_df.columns.difference(
        ["model", "class_id", "class_name"]
    )

    out_df[numeric_cols] = (
        out_df[numeric_cols].round(2)
    )

    out_path = (
            result_dir
            / f"finetune_f1_improvement_epoch0_{max_epoch}.csv"
    )

    out_df.to_csv(
        out_path,
        index=False
    )

    print("saved:", out_path)

    return out_df

if __name__ == "__main__":

    result_dir = Path(config.PROJECT_ROOT) / "results"

    models = [
        "ResNet50",
        "MobileNetV2",
        "ViT16"
    ]

    epochs = (
        [0]
        + list(range(10, 150, 10))
        + [150]
    )

    seeds = [
        12,
        123,
        1234
    ]

    classes = list(range(10))

    MODEL_COLORS = {
        "ResNet50": "tab:blue",
        "MobileNetV2": "tab:orange",
        "ViT16": "tab:green",
    }

    training_df = build_training_df(
        result_dir,
        models,
        epochs,
        seeds
    )

    real_df = build_real_df(
        result_dir,
        models,
        seeds
    )

    summary_50_df = save_metric_summary_csv(
        training_df=training_df,
        real_df=real_df,
        result_dir=result_dir,
        max_epoch=50,
    )

    # F1
    f1_summary = summarize_metric(
        training_df,
        metric="f1",
        include_epoch=True
    )

    real_f1_summary = summarize_metric(
        real_df,
        metric="f1",
        include_epoch=False
    )

    plot_metric(
        summary_df=f1_summary,
        real_summary_df=real_f1_summary,
        metric="f1",
        models=models,
        classes=classes,
        model_colors=MODEL_COLORS,
        save_path=(
            result_dir
            / "additional_finetune_each_epochs_f1.png"
        ),
    )

    # Accuracy
    acc_summary = summarize_metric(
        training_df,
        metric="accuracy",
        include_epoch=True
    )

    real_acc_summary = summarize_metric(
        real_df,
        metric="accuracy",
        include_epoch=False
    )

    plot_metric(
        summary_df=acc_summary,
        real_summary_df=real_acc_summary,
        metric="accuracy",
        models=models,
        classes=classes,
        model_colors=MODEL_COLORS,
        save_path=(
            result_dir
            / "additional_finetune_each_epochs_accuracy.png"
        ),
    )