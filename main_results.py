from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_confusion_matrix(csv_paths, label_to_class, title="", normalize="true", ax=None, add_colorbar=False,):

    labels = sorted(label_to_class.keys())
    class_names = [label_to_class[i] for i in labels]
    confusion_matrices = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Prediction CSV was not found: {csv_path}"
            )

        df = pd.read_csv(csv_path)

        required_columns = {"label", "preds"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing columns: {missing_columns}"
            )

        y_true = df["label"].to_numpy()
        y_pred = df["preds"].to_numpy()

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize=normalize,
        )

        confusion_matrices.append(cm.astype(float))

    # shape:
    # (number of seeds, number of classes, number of classes)
    confusion_matrices = np.stack(confusion_matrices, axis=0)

    cm_mean = np.mean(confusion_matrices, axis=0)
    cm_std = np.std(confusion_matrices, axis=0, ddof=1)

    created_fig = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(
        cm_mean,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize is not None else None,
    )

    if add_colorbar:
        fig.colorbar(im, ax=ax)

    ax.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="right",
    )

    fmt = ".2f"

    threshold = (
        np.nanmax(cm_mean) * 0.6
        if np.size(cm_mean)
        else 0
    )

    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            ax.text(
                j,
                i,
                format(cm_mean[i, j], fmt),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm_mean[i, j] > threshold
                    else "black"
                ),
                fontsize=14,
            )

    if created_fig:
        fig.tight_layout()

    return cm_mean, cm_std, class_names, im


def create_mean_std_results_table(
    seeds,
    train_dbs,
    model_names,
    test_settings=("REAL", "FAKE"),
):
    results_dir = config.PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for train_db in train_dbs:
        for test_setting in test_settings:
            for model_name in model_names:

                accuracy_values = []
                f1_values = []

                for seed in seeds:

                    csv_path = (
                        results_dir /
                        f"{model_name}_{train_db}_{test_setting}_seed_{seed}.csv"
                    )

                    df = pd.read_csv(csv_path)

                    y_true = df["label"]
                    y_pred = df["preds"]

                    accuracy_values.append(
                        accuracy_score(y_true, y_pred)
                    )

                    f1_values.append(
                        f1_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    )

                summary_rows.append(
                    {
                        "Train": train_db,
                        "Test": test_setting,
                        "Model": model_name,
                        "Accuracy": (
                            f"{np.mean(accuracy_values):.3f} ± "
                            f"{np.std(accuracy_values, ddof=1):.3f}"
                        ),
                        "F1": (
                            f"{np.mean(f1_values):.3f} ± "
                            f"{np.std(f1_values, ddof=1):.3f}"
                        ),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        results_dir / "performance_paper_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary_df


def main():

    # Plot averaged confusion matrices
    model_names = [
        conf.model_name
        for _, conf in EXPERIMENT_MODELS
    ]

    for train_db in DB:

        for test_setting in ["REAL", "FAKE"]:

            fig, axes = plt.subplots(
                1,
                len(model_names),
                figsize=(5 * len(model_names), 6),
                constrained_layout=True,
                sharey=True,
            )

            axes = np.atleast_1d(axes)

            mean_cms = {}
            std_cms = {}
            images = []

            for i, (ax, model_name) in enumerate(
                    zip(axes, model_names)
            ):
                csv_paths = [
                    config.PROJECT_ROOT
                    / "results"
                    / (
                        f"{model_name}_{train_db}_"
                        f"{test_setting}_seed_{seed}.csv"
                    )
                    for seed in seeds
                ]

                cm_mean, cm_std, class_names, im = (
                    plot_mean_confusion_matrix(
                        csv_paths=csv_paths,
                        label_to_class=config.label_to_class,
                        title=model_name,
                        normalize="true",
                        ax=ax,
                        add_colorbar=False,
                    )
                )

                mean_cms[model_name] = cm_mean
                std_cms[model_name] = cm_std
                images.append(im)

                if i != 0:
                    ax.set_ylabel("")
                    ax.tick_params(
                        axis="y",
                        left=False,
                        labelleft=False,
                    )

            fig.colorbar(
                images[0],
                ax=axes,
                fraction=0.025,
                pad=0.02,
                label="Mean proportion",
            )

            out_path = (
                    config.PROJECT_ROOT
                    / "results"
                    / (
                        f"class_heatmap_{train_db}_"
                        f"{test_setting}_mean.png"
                    )
            )

            fig.savefig(
                out_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(f"Saved confusion matrix: {out_path}")

    create_mean_std_results_table(
        seeds=seeds,
        train_dbs=DB,
        model_names=model_names,
    )