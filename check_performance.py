import config
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if __name__== "__main__":

    all_rows = []
    DB = list(config.CFG.keys())

    for db in DB:
        for m in config.MODELS:
            csv_path = config.PROJECT_ROOT / f"results/{m}_{db}_REAL.csv"
            df = pd.read_csv(csv_path)

            y_true = df["label"]
            y_pred = df["preds"]

            report_dict = classification_report(
                y_true, y_pred,
                digits=3, zero_division=0,
                output_dict=True
            )

            rep_df = pd.DataFrame(report_dict).T
            rep_df.index.name = "class"
            rep_df = rep_df.reset_index()

            rep_df["model"] = m
            rep_df["db"] = db
            rep_df[["precision", "recall", "f1-score"]] = rep_df[["precision", "recall", "f1-score"]].round(2)

            all_rows.append(rep_df)

    big_df = pd.concat(all_rows, ignore_index=True)

    MODEL_MARKERS = {
        "MobileNetV2": "o",
        "ResNet50": "^",
        "ViT16": "s",
    }

    db_list = sorted(big_df["db"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    DB_COLORS = {db: cmap(i) for i, db in enumerate(db_list)}

    METRIC_STYLES = {
        "precision": "-",
        "recall": "--",
        "f1-score": ":",
    }

    metrics = ["precision", "recall", "f1-score"]

    exclude_classes = {"accuracy", "macro avg", "weighted avg"}
    plot_df = big_df[~big_df["class"].isin(exclude_classes)].copy()
    class_order = sorted(plot_df["class"].unique().tolist())

    # ===== plotting =====
    fig, ax = plt.subplots(figsize=(19, 6))

    for db in db_list:
        for model in sorted(plot_df["model"].unique()):
            for met in metrics:
                sub = plot_df[(plot_df["db"] == db) & (plot_df["model"] == model)].copy()
                if sub.empty:
                    continue

                sub["class"] = pd.Categorical(sub["class"], categories=class_order, ordered=True)
                sub = sub.sort_values("class")

                ax.plot(
                    sub["class"].astype(str),
                    sub[met].astype(float),
                    color=DB_COLORS[db],
                    marker=MODEL_MARKERS.get(model, "o"),
                    linestyle=METRIC_STYLES[met],
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.9,
                )

    ax.set_ylabel("score", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=90, labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    ticks = ax.get_xticks()
    ticklabels = ax.get_xticklabels()

    new_labels = []
    for t in ticklabels:
        s = t.get_text()
        if s == "":
            new_labels.append("")
            continue
        k = int(float(s))
        new_labels.append(config.label_to_class.get(k, s))
    ax.set_xticklabels(new_labels, rotation=90)

    db_handles = [
        Line2D([0], [0], color=DB_COLORS[db], lw=3, label=f"DB: {db}")
        for db in db_list
    ]

    model_list = sorted(plot_df["model"].unique().tolist())
    model_handles = [
        Line2D([0], [0], color="black", marker=MODEL_MARKERS.get(m, "o"),
               linestyle="None", markersize=8, label=f"{m}")
        for m in model_list
    ]

    metric_handles = [
        Line2D([0], [0], color="black", linestyle=METRIC_STYLES[met], lw=2, label=f"{met}")
        for met in metrics
    ]

    db_label_map = {
        "FAKE1": "CIFAKE1",
        "FAKE2": "CIFAKE2",
        "REAL": "REAL",
    }

    x = 1.0
    db_handles = [
        Line2D(
            [0], [0],
            color=DB_COLORS[db],
            lw=3,
            label=db_label_map.get(db, db)
        )
        for db in db_list
    ]
    leg1 = ax.legend(handles=db_handles, loc="upper left", bbox_to_anchor=(x, 1.00), title="DB")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(x, 0.55), title="Model")
    ax.add_artist(leg2)
    ax.legend(handles=metric_handles, loc="upper left", bbox_to_anchor=(x, 0.25), title="Metric")

    plt.tight_layout()
    out_path = config.PROJECT_ROOT / "results" / f"performance_each_class.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

