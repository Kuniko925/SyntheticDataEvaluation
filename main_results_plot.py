import config
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__ == "__main__":

    all_rows = []

    DB = list(config.CFG.keys())
    seeds = [12, 123, 1234]

    for db in DB:
        for m in config.MODELS:
            for seed in seeds:

                csv_path = (
                    config.PROJECT_ROOT
                    / f"results/{m}_{db}_REAL_{seed}.csv"
                )

                df = pd.read_csv(csv_path)

                y_true = df["label"]
                y_pred = df["preds"]

                report_dict = classification_report(
                    y_true,
                    y_pred,
                    digits=3,
                    zero_division=0,
                    output_dict=True
                )

                classes = sorted(y_true.unique())

                for cls in classes:

                    # class-wise F1
                    f1 = report_dict[str(cls)]["f1-score"]

                    mask = (y_true == cls)
                    acc = (y_pred[mask] == cls).mean()

                    all_rows.append({
                        "db": db,
                        "model": m,
                        "seed": seed,
                        "class": cls,
                        "accuracy": acc,
                        "f1-score": f1,
                    })

    # ==========================================
    # DataFrame
    # ==========================================
    big_df = pd.DataFrame(all_rows)

    # 3 seeds: mean ± SD
    summary_df = (
        big_df
        .groupby(["db", "model", "class"])[
            ["accuracy", "f1-score"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    # ==========================================
    # Plot settings
    # ==========================================
    metrics = ["accuracy", "f1-score"]

    MODEL_MARKERS = {
        "MobileNetV2": "o",
        "ResNet50": "^",
        "ViT16": "s",
    }

    METRIC_STYLES = {
        "accuracy": "-",
        "f1-score": "--",
    }

    db_list = sorted(big_df["db"].unique().tolist())
    model_list = sorted(big_df["model"].unique().tolist())
    class_order = sorted(big_df["class"].unique().tolist())

    cmap = plt.get_cmap("tab10")
    DB_COLORS = {
        db: cmap(i)
        for i, db in enumerate(db_list)
    }

    # x軸 = class
    x_vals = np.arange(len(class_order))

    # ==========================================
    # Plot
    # ==========================================
    fig, ax = plt.subplots(figsize=(19, 6))

    for db in db_list:
        for model in model_list:
            for met in metrics:

                sub = summary_df[
                    (summary_df["db"] == db) &
                    (summary_df["model"] == model)
                ].copy()

                if sub.empty:
                    continue

                sub["class"] = pd.Categorical(
                    sub["class"],
                    categories=class_order,
                    ordered=True
                )

                sub = sub.sort_values("class")

                mean = sub[(met, "mean")].to_numpy(dtype=float)
                std = sub[(met, "std")].to_numpy(dtype=float)

                # 3 seeds
                ax.plot(
                    x_vals,
                    mean,
                    color=DB_COLORS[db],
                    marker=MODEL_MARKERS.get(model, "o"),
                    linestyle=METRIC_STYLES[met],
                    linewidth=1.8,
                    markersize=6,
                    alpha=0.95,
                )

                # ± SD
                ax.fill_between(
                    x_vals,
                    np.clip(mean - std, 0, 1),
                    np.clip(mean + std, 0, 1),
                    color=DB_COLORS[db],
                    alpha=0.08,
                    linewidth=0,
                )

    # ==========================================
    # Axis
    # ==========================================
    ax.set_xticks(x_vals)

    ax.set_xticklabels(
        [
            config.label_to_class.get(int(cls), str(cls))
            for cls in class_order
        ],
        rotation=90,
        fontsize=14
    )

    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0, 1.0)

    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # ==========================================
    # Legends
    # ==========================================
    db_label_map = {
        "FAKE1": "SDGen",
        "FAKE2": "EDMGen",
        "REAL": "REAL",
    }

    db_handles = [
        Line2D(
            [0], [0],
            color=DB_COLORS[db],
            lw=3,
            label=db_label_map.get(db, db)
        )
        for db in db_list
    ]

    model_handles = [
        Line2D(
            [0], [0],
            color="black",
            marker=MODEL_MARKERS.get(model, "o"),
            linestyle="None",
            markersize=8,
            label=model
        )
        for model in model_list
    ]

    metric_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=METRIC_STYLES[met],
            lw=2,
            label={
                "accuracy": "Accuracy",
                "f1-score": "F1"
            }[met]
        )
        for met in metrics
    ]

    legend_x = 1.0

    leg1 = ax.legend(
        handles=db_handles,
        loc="upper left",
        bbox_to_anchor=(legend_x, 1.00),
        title="DB"
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=model_handles,
        loc="upper left",
        bbox_to_anchor=(legend_x, 0.62),
        title="Model"
    )
    ax.add_artist(leg2)

    ax.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(legend_x, 0.30),
        title="Metric"
    )

    # ==========================================
    # Save
    # ==========================================
    plt.tight_layout()

    out_path = (
        config.PROJECT_ROOT
        / "results"
        / "performance_accuracy_f1_seeds.png"
    )

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()