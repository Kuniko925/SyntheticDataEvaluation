from __future__ import annotations
import pandas as pd
import numpy as np
import config
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def add_distance_to_centroid(
    centroids,
    points,
    class_col: str = "label",
    x_col: str = "embeddings x",
    y_col: str = "embeddings y",
    metric: str = "euclidean",   # "euclidean" or "squared"
    out_col: str = "dist_to_centroid",
) -> pd.DataFrame:
    cdf = pd.read_csv(centroids)
    pdf = pd.read_csv(points)
    merged = pdf.merge(cdf[[class_col, "r centroid x", "r centroid y"]], on=class_col, how="left")
    dx = merged[x_col] - merged["r centroid x"]
    dy = merged[y_col] - merged["r centroid y"]

    if metric == "euclidean":
        merged[out_col] = np.sqrt(dx * dx + dy * dy)
    elif metric == "squared":
        merged[out_col] = dx * dx + dy * dy
    else:
        raise ValueError("metric needs to be 'euclidean' or 'squared'")
    return merged

if __name__== "__main__":

    model_names = ['CLIP', 'DINOv2', 'DINOv3']
    reducer_names = ['UMAP', 'TSNE']
    DB = ['FAKE1', 'FAKE2']
    metric_colors = {"precision": "C0", "recall": "C1", "f1-score": "C2",}
    metric_markers = {"precision": "o", "recall": "^", "f1-score": "s"}
    db_alpha = {"FAKE1": 1.0, "FAKE2": 0.35}


    for model_name in model_names:
        for reducer_name in reducer_names:

            n_models = len(config.MODELS)
            fig, axes = plt.subplots(
                1, n_models,
                figsize=(6 * n_models, 5),
                sharex=True,
                sharey=True
            )
            if n_models == 1:
                axes = [axes]

            for i, m in enumerate(config.MODELS):
                ax = axes[i]

                for db in DB:
                    # --- Distance ---
                    df_distance = add_distance_to_centroid(
                        centroids=config.PROJECT_ROOT / f"results/dis_{db}_{model_name}_{reducer_name}_123.csv",
                        points=config.PROJECT_ROOT / f"results/embed_{db}_{model_name}_{reducer_name}_123.csv"
                    )
                    df_distance = df_distance[df_distance["rf"] == db]
                    df_distance = (
                        df_distance.groupby("label", as_index=False)["dist_to_centroid"]
                        .mean()
                        .rename(columns={"dist_to_centroid": "mean_dist_to_centroid"})
                    )

                    # --- Performance ---
                    filepath = config.PROJECT_ROOT / f"results/{m}_{db}_REAL_123.csv"
                    df_performance = pd.read_csv(filepath)

                    y_true = df_performance["label"]
                    y_pred = df_performance["preds"]

                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    df_report = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})

                    df_report_cls = df_report[df_report["label"].astype(str).str.fullmatch(r"\d+")].copy()
                    df_report_cls["label"] = df_report_cls["label"].astype(int)
                    df_report_cls = df_report_cls[["label", "precision", "recall", "f1-score", "support"]]
                    df_report_cls = df_report_cls.merge(df_distance, on="label", how="left")

                    x = df_report_cls["mean_dist_to_centroid"]
                    a = db_alpha[db]

                    for metric in ["precision", "recall", "f1-score"]:
                        ax.scatter(
                            x, df_report_cls[metric],
                            color=metric_colors[metric],
                            alpha=a,
                            marker=metric_markers[metric],
                            edgecolors="none",
                            s=80
                        )

                    if db == "FAKE1":
                        for _, r in df_report_cls.iterrows():
                            cls_name = str(config.label_to_class.get(r["label"], r["label"]))
                            ax.text(
                                r["mean_dist_to_centroid"],
                                r["f1-score"],
                                cls_name,
                                fontsize=12
                            )

                ax.set_title(f"{m}", fontsize=16)
                ax.set_xlabel("Distance from Centroid", fontsize=16)
                if i == 0:
                    ax.set_ylabel("Performance", fontsize=16)
                ax.set_ylim(-0.02, 1.02)

            metric_handles = [
                Line2D([0], [0], marker=metric_markers["precision"], linestyle="None",
                       markerfacecolor=metric_colors["precision"], markeredgecolor="none", label="precision"),
                Line2D([0], [0], marker=metric_markers["recall"], linestyle="None",
                       markerfacecolor=metric_colors["recall"], markeredgecolor="none", label="recall"),
                Line2D([0], [0], marker=metric_markers["f1-score"], linestyle="None",
                       markerfacecolor=metric_colors["f1-score"], markeredgecolor="none", label="f1-score"),
            ]
            db_handles = [
                Line2D([0], [0], marker="o", linestyle="None", color="k",
                       alpha=db_alpha["FAKE1"], label="FAKE1"),
                Line2D([0], [0], marker="o", linestyle="None", color="k",
                       alpha=db_alpha["FAKE2"], label="FAKE2"),
            ]

            fig.tight_layout(rect=[0, 0.06, 1, 1])
            fig.legend(
                handles=metric_handles + db_handles,
                loc="lower center",
                ncol=5,
                bbox_to_anchor=(0.5, 0.02),
            )
            save_filepath = config.PROJECT_ROOT / f"results/{model_name}_{reducer_name}_prec_recall_f1_FAKE1_FAKE2_123.png"
            fig.savefig(save_filepath, bbox_inches="tight")
            plt.close(fig)

