from __future__ import annotations
import pandas as pd
import numpy as np
import config
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

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

    for db in DB:
        for model_name in model_names:
            for reducer_name in reducer_names:
                # Distance
                df_distance = add_distance_to_centroid(
                    centroids= config.PROJECT_ROOT / f"results/dis_{db}_{model_name}_{reducer_name}.csv",
                    points= config.PROJECT_ROOT / f"results/embed_{db}_{model_name}_{reducer_name}.csv"
                )
                df_distance = df_distance[df_distance['rf'] == 'FAKE']
                df_distance = (
                    df_distance.groupby("label", as_index=False)["dist_to_centroid"]
                    .mean()
                    .rename(columns={"dist_to_centroid": "mean_dist_to_centroid"})
                )

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

                    # Performance
                    filepath = config.PROJECT_ROOT / f"results/{m}_{db}_REAL.csv"
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

                    ax.scatter(x, df_report_cls["precision"], label="precision")
                    ax.scatter(x, df_report_cls["recall"], label="recall")
                    ax.scatter(x, df_report_cls["f1-score"], label="f1-score")

                    for _, r in df_report_cls.iterrows():
                        cls_name = str(config.label_to_class.get(r["label"], r["label"]))
                        ax.text(
                            r["mean_dist_to_centroid"],
                            r["f1-score"],
                            cls_name,
                            fontsize=10
                        )

                    ax.set_title(f"{m}")
                    ax.set_xlabel("Mean distance to centroid")
                    if i == 0:
                        ax.set_ylabel("Performance")

                    ax.set_ylim(-0.02, 1.02)

                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", ncol=3)

                fig.suptitle("Performance vs centroid distance", y=1.03)
                fig.tight_layout()

                save_filepath = config.PROJECT_ROOT / f"results/{db}_{model_name}_{reducer_name}_prec_recall_f1.png"
                fig.savefig(save_filepath, bbox_inches="tight")
                plt.close(fig)



